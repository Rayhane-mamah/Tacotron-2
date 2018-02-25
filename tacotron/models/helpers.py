import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper
from .modules import stop_token_projection


# Adapted from tf.contrib.seq2seq.GreedyEmbeddingHelper
class TacoTestHelper(Helper):
	def __init__(self, batch_size, output_dim, r=1):
		with tf.name_scope('TacoTestHelper'):
			self._batch_size = batch_size
			self._output_dim = output_dim

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def sample_ids_shape(self):
		return tf.TensorShape([])

	@property
	def sample_ids_dtype(self):
		return np.int32

	def initialize(self, name=None):
		return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

	def sample(self, time, outputs, state, name=None):
		return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

	def next_inputs(self, time, cell_outputs, state, LSTM_output, sample_ids, name=None):
		'''Stop on EOS. Otherwise, pass the last output as the next input and pass through state.'''
		with tf.name_scope('TacoTestHelper'):
			#At inference time, stop_error = None
			stop_error = None # we don't need it

			context = state.attention # Get context vector
			#finished = tf.reduce_all(tf.equal(outputs, self._end_token), axis=1)

			#Predict if the encoder should stop (dynamic end token)
			concat = tf.concat([LSTM_output, context], axis=-1)
			scalar = tf.squeeze(stop_token_projection(concat, activation=tf.nn.sigmoid), [1])
			finished = tf.cast(tf.round(scalar), tf.bool)

			# Feed last output frame as next input. outputs is [N, output_dim * r]
			next_inputs = cell_outputs
			next_state = state
			return (finished, next_inputs, next_state, stop_error)


class TacoTrainingHelper(Helper):
	def __init__(self, inputs, targets, output_dim, r=1):
		# inputs is [N, T_in], targets is [N, T_out, D]
		with tf.name_scope('TacoTrainingHelper'):
			self._batch_size = tf.shape(inputs)[0]
			self._output_dim = output_dim

			# Feed every r-th target frame as input
			self._targets = targets[:, r-1::r, :]

			# Use full length for every target because we don't want to mask the padding frames
			num_steps = tf.shape(self._targets)[1]
			self._lengths = tf.tile([num_steps], [self._batch_size])

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def sample_ids_shape(self):
		return tf.TensorShape([])

	@property
	def sample_ids_dtype(self):
		return np.int32

	def initialize(self, name=None):
		return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

	def sample(self, time, outputs, state, name=None):
		return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

	def next_inputs(self, time, cell_outputs, state, LSTM_output, sample_ids, name=None):
		with tf.name_scope(name or 'TacoTrainingHelper'):
			context = state.attention #Get context vector
			finished = (time + 1 >= self._lengths) #return true finished

			#Compute model prediction to stop token
			concat = tf.concat([LSTM_output, context], axis=-1)
			finished_p = tf.squeeze(stop_token_projection(concat), [1])

			#Compute the stop token error for infer time
			stop_error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(finished, tf.float32), 
																				logits=finished_p))
			next_inputs = self._targets[:, time, :] #teacher-forcing: return true frame
			next_state = state
			return (finished, next_inputs, next_state, stop_error)


def _go_frames(batch_size, output_dim):
	'''Returns all-zero <GO> frames for a given batch size and output dimension'''
	return tf.tile([[0.0]], [batch_size, output_dim])

