import tensorflow as tf 
from utils.symbols import symbols
from utils.infolog import log
from .helpers import TacoTrainingHelper, TacoTestHelper
from .modules import *
from models.zoneout_LSTM import ZoneoutLSTMCell
from tensorflow.contrib.seq2seq import dynamic_decode, BasicDecoder, BahdanauAttention
from .Architecture_wrappers import TacotronEncoderCell, TacotronDecoderCell
from .attention import LocationSensitiveAttention
from .custom_decoder import CustomDecoder


class Tacotron():
	def __init__(self, hparams):
		self._hparams = hparams

	def initialize(self, inputs, input_lengths, mel_targets=None, stop_token_targets=None, gta=False):
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
		if mel_targets is None and stop_token_targets is not None:
			raise ValueError('no mel targets were provided but token_targets were given')
		if mel_targets is not None and stop_token_targets is None:
			raise ValueError('Mel targets are provided without corresponding token_targets')

		with tf.variable_scope('inference') as scope:
			is_training = mel_targets is not None and not gta
			batch_size = tf.shape(inputs)[0]
			hp = self._hparams

			# Embeddings ==> [batch_size, sequence_length, embedding_dim]
			embedding_table = tf.get_variable(
				'inputs_embedding', [len(symbols), hp.embedding_dim], dtype=tf.float32,
				initializer=tf.contrib.layers.xavier_initializer(uniform=False))
			embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)


			#Encoder Cell ==> [batch_size, encoder_steps, encoder_lstm_units]
			encoder_cell = TacotronEncoderCell(
				EncoderConvolutions(is_training, kernel_size=hp.enc_conv_kernel_size,
					channels=hp.enc_conv_channels, scope='encoder_convolutions'),
				EncoderRNN(is_training, size=hp.encoder_lstm_units,
					zoneout=hp.zoneout_rate, scope='encoder_LSTM'))

			encoder_outputs = encoder_cell(embedded_inputs, input_lengths)

			#For shape visualization purpose
			enc_conv_output_shape = encoder_cell.conv_output_shape


			#Decoder Parts
			#Attention Decoder Prenet
			prenet = Prenet(is_training, layer_sizes=hp.prenet_layers, scope='decoder_prenet')
			#Attention Mechanism
			attention_mechanism = LocationSensitiveAttention(hp.attention_dim, encoder_outputs)
			#Decoder LSTM Cells
			decoder_lstm = DecoderRNN(is_training, layers=hp.decoder_layers,
				size=hp.decoder_lstm_units, zoneout=hp.zoneout_rate, scope='decoder_lstm')
			#Frames Projection layer
			frame_projection = FrameProjection(hp.num_mels * hp.outputs_per_step, scope='linear_transform')
			#<stop_token> projection layer
			stop_projection = StopProjection(is_training, scope='stop_token_projection')


			#Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
			decoder_cell = TacotronDecoderCell(
				prenet,
				attention_mechanism,
				decoder_lstm,
				frame_projection,
				stop_projection)


			#Define the helper for our decoder
			if (is_training or gta) == True:
				self.helper = TacoTrainingHelper(inputs, mel_targets, hp.num_mels, hp.outputs_per_step)
			else:
				self.helper = TacoTestHelper(batch_size, hp.num_mels, hp.outputs_per_step)


			#initial decoder state
			decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)


			#Decode
			(frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
				CustomDecoder(decoder_cell, self.helper, decoder_init_state),
				impute_finished=hp.impute_finished,
				maximum_iterations=hp.max_iters)


			# Reshape outputs to be one output per entry 
			#==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
			decoder_output = tf.reshape(frames_prediction, [batch_size, -1, hp.num_mels])
			stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])


			#Postnet
			postnet = Postnet(is_training, kernel_size=hp.postnet_kernel_size, 
				channels=hp.postnet_channels, scope='postnet_convolutions')

			#Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
			residual = postnet(decoder_output)

			#Project residual to same dimension as mel spectrogram 
			#==> [batch_size, decoder_steps * r, num_mels]
			residual_projection = FrameProjection(hp.num_mels, scope='postnet_projection')
			projected_residual = residual_projection(residual)


			#Compute the mel spectrogram
			mel_outputs = decoder_output + projected_residual

			#Grab alignments from the final decoder state
			alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

			self.inputs = inputs
			self.input_lengths = input_lengths
			self.decoder_output = decoder_output
			self.alignments = alignments
			self.stop_token_prediction = stop_token_prediction
			self.stop_token_targets = stop_token_targets
			self.mel_outputs = mel_outputs
			self.mel_targets = mel_targets
			log('Initialized Tacotron model. Dimensions: ')
			log('  embedding:                {}'.format(embedded_inputs.shape))
			log('  enc conv out:             {}'.format(enc_conv_output_shape))
			log('  encoder out:              {}'.format(encoder_outputs.shape))
			log('  decoder out:              {}'.format(decoder_output.shape))
			log('  residual out:             {}'.format(residual.shape))
			log('  projected residual out:   {}'.format(projected_residual.shape))
			log('  mel out:                  {}'.format(mel_outputs.shape))
			log('  <stop_token> out:         {}'.format(stop_token_prediction.shape))


	def add_loss(self):
		'''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
		with tf.variable_scope('loss') as scope:
			hp = self._hparams

			# Compute loss of predictions before postnet
			before = tf.losses.mean_squared_error(self.mel_targets, self.decoder_output)
			# Compute loss after postnet
			after = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)
			#Compute <stop_token> loss (for learning dynamic generation stop)
			stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
				labels=self.stop_token_targets,
				logits=self.stop_token_prediction))

			# Get all trainable variables
			all_vars = tf.trainable_variables()
			# Compute the regularization term
			regularization = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
				if not('bias' in v.name or 'Bias' in v.name)]) * hp.reg_weight

			# Compute final loss term
			self.before_loss = before
			self.after_loss = after
			self.stop_token_loss = stop_token_loss
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