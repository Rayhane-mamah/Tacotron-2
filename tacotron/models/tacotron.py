import tensorflow as tf 
from utils.symbols import symbols
from utils.infolog import log
from .helpers import TacoTrainingHelper, TacoTestHelper
from .modules import *
from models.zoneout_LSTM import ZoneoutLSTMCell
from tensorflow.contrib.seq2seq import AttentionWrapper, LuongAttention
from .rnn_wrappers import *
from tensorflow.contrib.rnn import MultiRNNCell, OutputProjectionWrapper
from .attention import LocationSensitiveAttention
from .custom_decoder import CustomDecoder
from .dynamic_decoder import dynamic_decode


class Tacotron():
	def __init__(self, hparams):
		self._hparams = hparams

	def initialize(self, inputs, input_lengths, mel_targets=None, gta=False):
		"""
		Initializes the model for inference

		sets "mel_outputs" and "alignments" fields.

		Args:
			- inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
			  steps in the input time series, and values are character IDs
			- input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
			of each sequence in inputs.
			- mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
			of steps in the output time series, M is num_mels, and values are entries in the mel
			spectrogram. Only needed for training.
		"""
		with tf.variable_scope('inference') as scope:
			is_training = mel_targets is not None and not gta
			batch_size = tf.shape(inputs)[0]
			hp = self._hparams

			# Embeddings
			embedding_table = tf.get_variable(
				'inputs_embedding', [len(symbols), hp.embedding_dim], dtype=tf.float32,
				initializer=tf.contrib.layers.xavier_initializer())
			embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)

			#Encoder
			enc_conv_outputs = enc_conv_layers(embedded_inputs, is_training,
				kernel_size=hp.enc_conv_kernel_size, channels=hp.enc_conv_channels)    
			#Paper doesn't specify what to do with final encoder state
			#So we will simply drop it
			encoder_outputs, encoder_states = bidirectional_LSTM(enc_conv_outputs, input_lengths,
				'encoder_LSTM', is_training=is_training, size=hp.encoder_lstm_units,
				zoneout=hp.zoneout_rate)     

			#Attention
			attention_cell = AttentionWrapper(
				DecoderPrenetWrapper(ZoneoutLSTMCell(hp.attention_dim, is_training, #Separate LSTM for attention mechanism
					zoneout_factor_cell=hp.zoneout_rate,							#based on original tacotron architecture
					zoneout_factor_output=hp.zoneout_rate), is_training),
				LocationSensitiveAttention(hp.attention_dim, encoder_outputs),
				alignment_history=True,
				output_attention=False,
				name='attention_cell')

			#Concat Prenet output with context vector
			concat_cell = ConcatPrenetAndAttentionWrapper(attention_cell)

			#Decoder layers (attention pre-net + 2 unidirectional LSTM Cells)
			decoder_cell = unidirectional_LSTM(concat_cell, is_training,
				layers=hp.decoder_layers, size=hp.decoder_lstm_units,
				zoneout=hp.zoneout_rate)

			#Concat LSTM output with context vector
			concat_decoder_cell = ConcatLSTMOutputAndAttentionWrapper(decoder_cell)

			#Projection to mel-spectrogram dimension (times number of outputs per step) (linear transformation)
			output_cell = OutputProjectionWrapper(concat_decoder_cell, hp.num_mels * hp.outputs_per_step)

			#Define the helper for our decoder
			if (is_training or gta) == True:
				self.helper = TacoTrainingHelper(inputs, mel_targets, hp.num_mels, hp.outputs_per_step)
			else:
				self.helper = TacoTestHelper(batch_size, hp.num_mels, hp.outputs_per_step)

			#We'll only limit decoder time steps during inference (consult hparams.py to modify the value)
			max_iterations = None if is_training else hp.max_iters

			#initial decoder state
			decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

			#Decode
			(decoder_output, _), final_decoder_state, self.stop_token_loss = dynamic_decode(
				CustomDecoder(output_cell, self.helper, decoder_init_state),
				impute_finished=hp.impute_finished,
				maximum_iterations=max_iterations)

			# Reshape outputs to be one output per entry 
			decoder_output = tf.reshape(decoder_output, [batch_size, -1, hp.num_mels])

			#Compute residual using post-net
			residual = postnet(decoder_output, is_training,
				kernel_size=hp.postnet_kernel_size, channels=hp.postnet_channels)

			#Project residual to same dimension as mel spectrogram
			projected_residual = projection(residual,
				shape=hp.num_mels,
				scope='residual_projection')

			#Compute the mel spectrogram
			mel_outputs = decoder_output + projected_residual

			#Grab alignments from the final decoder state
			alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

			self.inputs = inputs
			self.input_lengths = input_lengths
			self.decoder_output = decoder_output
			self.alignments = alignments
			self.mel_outputs = mel_outputs
			self.mel_targets = mel_targets
			log('Initialized Tacotron model. Dimensions: ')
			log('  embedding:                {}'.format(embedded_inputs.shape))
			log('  enc conv out:             {}'.format(enc_conv_outputs.shape))
			log('  encoder out:              {}'.format(encoder_outputs.shape))
			log('  decoder out:              {}'.format(decoder_output.shape))
			log('  residual out:             {}'.format(residual.shape))
			log('  projected residual out:   {}'.format(projected_residual.shape))
			log('  mel out:                  {}'.format(mel_outputs.shape))


	def add_loss(self):
		'''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
		with tf.variable_scope('loss') as scope:
			hp = self._hparams

			# Compute loss of predictions before postnet
			before = tf.losses.mean_squared_error(self.mel_targets, self.decoder_output)
			# Compute loss after postnet
			after = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)

			# Get all trainable variables
			all_vars = tf.trainable_variables()
			# Compute the regularization term
			regularization = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
				if not('bias' in v.name or 'Bias' in v.name)]) * hp.reg_weight

			# Compute final loss term
			self.before_loss = before
			self.after_loss = after
			self.regularization_loss = regularization

			self.loss = self.before_loss + self.after_loss + self.stop_token_loss + self.regularization_loss 

	def add_optimizer(self, global_step):
		'''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

		Args:
			global_step: int32 scalar Tensor representing current global step in training
		'''
		with tf.variable_scope('optimizer') as scope:
			hp = self._hparams
			if hp.decay_learning_rate:
				self.decay_steps = hp.decay_steps
				self.decay_rate = hp.decay_rate
				self.learning_rate = self._learning_rate_decay(hp.initial_learning_rate, global_step)
			else:
				self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)

			optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2, hp.adam_epsilon)
			gradients, variables = zip(*optimizer.compute_gradients(self.loss))
			self.gradients = gradients
			#Clip the gradients to avoid rnn gradient explosion
			clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

			# Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
			# https://github.com/tensorflow/tensorflow/issues/1122
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
					global_step=global_step)

	def _learning_rate_decay(self, init_lr, global_step):
		# Exponential decay
		# We won't drop learning rate below 10e-5
		hp = self._hparams
		step = tf.cast(global_step + 1, dtype=tf.float32)
		lr = tf.train.exponential_decay(init_lr, 
			global_step - self.decay_steps + 1, 
			self.decay_steps, 
			self.decay_rate,
			name='exponential_decay')
		return tf.maximum(hp.final_learning_rate, lr)