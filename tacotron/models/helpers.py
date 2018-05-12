import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper
from hparams import hparams


class TacoTestHelper(Helper):
	def __init__(self, batch_size, output_dim, r):
		with tf.name_scope('TacoTestHelper'):
			self._batch_size = batch_size
			self._output_dim = output_dim
			self._reduction_factor = r

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def token_output_size(self):
		return self._reduction_factor

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

	def next_inputs(self, time, outputs, state, sample_ids, stop_token_prediction, name=None):
		'''Stop on EOS. Otherwise, pass the last output as the next input and pass through state.'''
		with tf.name_scope('TacoTestHelper'):
			#A sequence is finished when the output probability is > 0.5
			finished = tf.cast(tf.round(stop_token_prediction), tf.bool)

			#Since we are predicting r frames at each step, two modes are 
			#then possible:
			#	Stop when the model outputs a p > 0.5 for any frame between r frames (Recommended)
			#	Stop when the model outputs a p > 0.5 for all r frames (Safer)
			#Note:
			#	With enough training steps, the model should be able to predict when to stop correctly
			#	and the use of stop_at_any = True would be recommended. If however the model didn't
			#	learn to stop correctly yet, (stops too soon) one could choose to use the safer option 
			#	to get a correct synthesis
			if hparams.stop_at_any:
				finished = tf.reduce_any(finished) #Recommended
			else:
				finished = tf.reduce_all(finished) #Safer option
			
			# Feed last output frame as next input. outputs is [N, output_dim * r]
			next_inputs = outputs[:, -self._output_dim:]
			next_state = state
			return (finished, next_inputs, next_state)


class TacoTrainingHelper(Helper):
	def __init__(self, batch_size, targets, stop_targets, output_dim, r, ratio, gta):
		# inputs is [N, T_in], targets is [N, T_out, D]
		with tf.name_scope('TacoTrainingHelper'):
			self._batch_size = batch_size
			self._output_dim = output_dim
			self._reduction_factor = r
			self._ratio = tf.convert_to_tensor(ratio)
			self.gta = gta

			# Feed every r-th target frame as input
			self._targets = targets[:, r-1::r, :]

			if not gta:
				# Detect finished sequence using stop_targets
				self._stop_targets = stop_targets[:, r-1::r]
			else:
				# GTA synthesis
				self._lengths = tf.tile([tf.shape(self._targets)[1]], [self._batch_size])

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def token_output_size(self):
		return self._reduction_factor

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

	def next_inputs(self, time, outputs, state, sample_ids, stop_token_prediction, name=None):
		with tf.name_scope(name or 'TacoTrainingHelper'):
			if not self.gta:
				#mark sequences where stop_target == 1 as finished (for case of imputation)
				finished = tf.equal(self._stop_targets[:, time], [1.])
			else:
				#GTA synthesis stop
				finished = (time + 1 >= self._lengths)

			next_inputs = tf.cond(
				tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), self._ratio),
				lambda: self._targets[:, time, :], #Teacher-forcing: return true frame
				lambda: outputs[:,-self._output_dim:])

			#Update the finished state
			next_state = state.replace(finished=tf.cast(tf.reshape(finished, [-1, 1]), tf.float32))
			return (finished, next_inputs, next_state)


def _go_frames(batch_size, output_dim):
	'''Returns all-zero <GO> frames for a given batch size and output dimension'''
	return tf.tile([[0.0]], [batch_size, output_dim])