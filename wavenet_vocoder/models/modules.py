import numpy as np 
import tensorflow as tf 
from wavenet_vocoder.util import sequence_mask
from .mixture import discretized_mix_logistic_loss
from .gaussian import gaussian_maximum_likelihood_estimation_loss

class Embedding:
	"""Embedding class for global conditions.
	"""
	def __init__(self, num_embeddings, embedding_dim, std=0.1, name='gc_embedding'):
		#Create embedding table
		self.embedding_table = tf.get_variable(name,
			[num_embeddings, embedding_dim], dtype=tf.float32,
			initializer=tf.truncated_normal_initializer(mean=0., stddev=std))

	def __call__(self, inputs):
		#Do the actual embedding
		return tf.nn.embedding_lookup(self.embedding_table, inputs)

class ReluActivation:
	"""Simple class to wrap relu activation function in class for later call.
	"""
	def __init__(self, name=None):
		self.name = name

	def __call__(self, inputs):
		return tf.nn.relu(inputs, name=self.name)


class LeakyReluActivation:
	'''Simple class to wrap leaky relu activation function in class for later call.
	'''
	def __init__(self, alpha=0.3, name=None):
		self.alpha = alpha
		self.name = name

	def __call__(self, inputs):
		return tf.nn.leaky_relu(inputs, alpha=self.alpha, name=self.name)


class Conv1d1x1(tf.layers.Conv1D):
	"""Extend tf.layers.Conv1D for dilated layers convolutions.
	"""
	def __init__(self, in_channels, filters, kernel_size=1, padding=None, dilation=1, use_bias=True, name='Conv1d1x1'):
		with tf.variable_scope(name):
			#Create variables
			kernel_shape = (kernel_size, in_channels, filters)
			self.kernel = tf.get_variable(
				name='kernel_{}'.format(name),
				shape=kernel_shape,
				dtype=tf.float32
				)

			if use_bias:
				self.bias = tf.get_variable(
					name='bias_{}'.format(name),
					shape=(filters, ),
					initializer=tf.zeros_initializer(),
					dtype=tf.float32)

			self.filters = filters
			self.in_channels = in_channels
			self.dilation_rate = dilation
			self.use_bias = use_bias
			self.paddings = padding
			self.scope = name

			#reshape weight and get kernel_width (only used for incremental step)
			self.kw = self.kernel.shape[0]
			self.linearized_weight = self._get_linearized_weight()

	def _get_linearized_weight(self):
		#layers.Conv1D
		if tf.shape(self.kernel) == (self.filters, self.in_channels, self.kw):
			#[filters, in, kw]
			weight = tf.transpose(self.kernel, [2, 1, 0])
		else:
			#[kw, in, filters]
			weight = self.kernel

		#[kw, in, filters]
		assert weight.shape == (self.kw, self.in_channels, self.filters)
		return tf.cast(tf.reshape(weight, [-1, self.filters]), dtype=tf.float32)

	def set_mode(self, is_training):
		self.training = is_training

	def _to_dilation(self, inputs):
		'''Pad and reshape inputs by dilation rate.

		Used to perfrom 1D dilation convolution.
		'''
		if self.paddings is not None: #dilated conv
			assert isinstance(self.paddings, int)
			# inputs_padded = tf.pad(inputs, [[0, 0], [0, 0], [self.paddings, 0]], "CONSTANT")

			#inputs are channels first
			inputs_shape = tf.shape(inputs)
			width = inputs_shape[-1]
			channels = inputs_shape[1]
			#width_pad = inputs_shape[-1]
			width_pad = tf.cast(self.dilation_rate * (tf.ceil(tf.cast(width + self.dilation_rate, tf.float32) / self.dilation_rate) + tf.cast(self.kw - 2, tf.float32)), 
				tf.int32)
			pad_left = width_pad - width

			# pad_left = self.dilation_rate - 1 - (width + self.dilation_rate - 1) % self.dilation_rate
			# width_pad = width + pad_left

			with tf.control_dependencies([tf.assert_equal(width_pad % self.dilation_rate, 0)]):
				width_pad = tf.identity(width_pad)

			dilation_shape = (width_pad // self.dilation_rate, inputs_shape[0] * self.dilation_rate, channels) #-1 refers to batch_size * dilation_rate
			inputs_padded = tf.pad(inputs, [[0, 0], [0, 0], [pad_left, 0]], "CONSTANT")
			#[width_pad, batch_size, channels]
			inputs_transposed = tf.transpose(inputs_padded, [2, 0, 1])
			#[width_pad / dilation_rate, batch_size * dilation_rate, channels]
			inputs_reshaped = tf.reshape(inputs_transposed, dilation_shape)
			#[batch_size * dilation_rate, width_pad / dilation_rate, channels]
			outputs = tf.transpose(inputs_reshaped, [1, 0, 2])

		else: #Simple channels last convolution
			outputs = tf.transpose(inputs, [0, 2, 1])

		return outputs

	def _from_dilation(self, inputs, crop):
		'''Remove paddings and reshape to 1d signal.

		Used after 1D dilation convolution.
		'''
		if self.paddings is not None: #dilated conv
			assert isinstance(self.paddings, int)
			#inputs: [batch_size * dilation_rate, width_pad / dilation_rate, channels]
			inputs_shape = tf.shape(inputs)
			batch_size = inputs_shape[0] / self.dilation_rate
			width_pad = inputs_shape[1] * self.dilation_rate
			channels = inputs_shape[-1]
			new_shape = (width_pad, -1, channels) #-1 refers to batch_size

			#[width_pad / dilation_rate, batch_size * dilation_rate, channels]
			inputs_transposed = tf.transpose(inputs, [1, 0, 2])
			#[width_pad, batch_size, channels]
			inputs_reshaped = tf.reshape(inputs_transposed, new_shape)
			#[batch_size, channels, width_pad]
			outputs = tf.transpose(inputs_reshaped, [1, 2, 0])
			#[batch_size, channels, width]
			cropped = tf.slice(outputs, [0, 0, crop], [-1, -1, -1])

		else: #Simple channels last convolution
			cropped = tf.transpose(inputs, [0, 2, 1])

		return cropped
		

	def __call__(self, inputs):
		'''During this call, we change to channel last scheme for a better generalization and easier bias computation
		'''
		with tf.variable_scope(self.scope):
			#Reshape to dilated conv mode (if this instance is of a dilated convolution)
			inputs_ = self._to_dilation(inputs)

			outputs_ = tf.nn.conv1d(inputs_, self.kernel,
				stride=1, padding='VALID', data_format='NWC')

			if self.use_bias:
				outputs_ = tf.nn.bias_add(outputs_, self.bias)

			#Reshape back ((if this instance is of a dilated convolution))
			diff = tf.shape(outputs_)[1] * self.dilation_rate - tf.shape(inputs)[-1]
			outputs = self._from_dilation(outputs_, crop=diff)

			#Make sure that outputs have same time steps as inputs
			#[batch_size, channels(filters), width]
			with tf.control_dependencies([tf.assert_equal(tf.shape(outputs)[-1], tf.shape(inputs)[-1])]):
				outputs = tf.identity(outputs, name='output_equal_input_time_assert')

			return outputs

	def incremental_step(self, inputs, convolution_queue=None):
		'''At sequential inference times:
		we adopt fast wavenet convolution queues by saving precomputed states for faster generation

		inputs: [batch_size, time_length, channels] ('NWC')! Channels last!
		'''
		with tf.variable_scope(self.scope):
			#input: [batch_size, time_length, channels]
			if self.training: 
				raise RuntimeError('incremental_step only supports eval mode')

			batch_size = tf.shape(inputs)[0]
			#Fast dilation
			#Similar to using tf FIFOQueue to schedule states of dilated convolutions
			if self.kw > 1:
				#shift queue (remove first element for following append)
				convolution_queue = convolution_queue[:, 1:, :]
				
				#append next input
				convolution_queue = tf.concat([convolution_queue, tf.expand_dims(inputs[:, -1, :], axis=1)], axis=1)

				inputs = convolution_queue
				if self.dilation_rate > 1:
					inputs = inputs[:, 0::self.dilation_rate, :]

			#Compute step prediction
			output = tf.matmul(tf.reshape(inputs, [batch_size, -1]), self.linearized_weight)
			if self.use_bias:
				output = tf.nn.bias_add(output, self.bias)

			#[batch_size, 1(time_step), channels(filters)]
			return tf.reshape(output, [batch_size, 1, self.filters]), convolution_queue

	def clear_queue(self):
		self.convolution_queue = None

def _conv1x1_forward(conv, x, is_incremental):
	"""conv1x1 step
	"""
	if is_incremental:
		output, _ = conv.incremental_step(x)
		return output
	else:
		return conv(x)

class ResidualConv1dGLU():
	'''Residual dilated conv1d + Gated Linear Unit
	'''

	def __init__(self, residual_channels, gate_channels, kernel_size,
			skip_out_channels=None, cin_channels=-1, gin_channels=-1,
			dropout=1 - .95, padding=None, dilation=1, causal=True,
			use_bias=True, name='ResidualConv1dGLU'):
		self.dropout = dropout

		if skip_out_channels is None:
			skip_out_channels = residual_channels

		if padding is None:
			#No future time stamps available
			if causal:
				padding = (kernel_size - 1) * dilation
			else:
				padding = (kernel_size - 1) // 2 * dilation

		self.causal = causal

		self.conv = Conv1d1x1(residual_channels, gate_channels, kernel_size,
			padding=padding, dilation=dilation, use_bias=use_bias, name='residual_block_conv_{}'.format(name))

		#Local conditioning
		if cin_channels > 0:
			self.conv1x1c = Conv1d1x1(cin_channels, gate_channels,
				use_bias=use_bias, name='residual_block_cin_conv_{}'.format(name))
		else:
			self.conv1x1c = None

		#Global conditioning
		if gin_channels > 0:
			self.conv1x1g = Conv1d1x1(gin_channels, gate_channels, 
				use_bias=use_bias, name='residual_block_gin_conv_{}'.format(name))
		else:
			self.conv1x1g = None

		gate_out_channels = gate_channels // 2
		self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, use_bias=use_bias, name='residual_block_out_conv_{}'.format(name))
		self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, use_bias=use_bias, name='residual_block_skip_conv_{}'.format(name))

	def set_mode(self, is_training):
		for conv in [self.conv, self.conv1x1c, self.conv1x1g, self.conv1x1_out, self.conv1x1_skip]:
			try:
				conv.set_mode(is_training)
			except AttributeError:
				pass

	def __call__(self, x, c=None, g=None):
		x, s, _ = self.step(x, c, g, False)
		return (x, s)

	def incremental_step(self, x, c=None, g=None, queue=None):
		return self.step(x, c, g, True, queue=queue)

	def step(self, x, c, g, is_incremental, queue=None):
		'''

		Args:
			x: Tensor [batch_size, channels, time_length]
			c: Tensor [batch_size, c_channels, time_length]. Local conditioning features
			g: Tensor [batch_size, g_channels, time_length], global conditioning features
			is_incremental: Boolean, whether incremental mode is on
		Returns:
			Tensor output
		'''
		residual = x
		x = tf.layers.dropout(x, rate=self.dropout, training=not is_incremental)
		if is_incremental:
			splitdim = -1
			x, queue = self.conv.incremental_step(x, queue)
		else:
			splitdim = 1
			x = self.conv(x)
			#Remove future time steps
			x = x[:, :, :tf.shape(residual)[-1]] if self.causal else x

		a, b = tf.split(x, num_or_size_splits=2, axis=splitdim)

		#local conditioning
		if c is not None:
			assert self.conv1x1c is not None
			c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
			ca, cb = tf.split(c, num_or_size_splits=2, axis=splitdim)
			a, b = a + ca, b + cb

		#global conditioning
		if g is not None:
			assert self.conv1x1g is not None
			g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
			ga, gb = tf.split(g, num_or_size_splits=2, axis=splitdim)
			a, b = a + ga, b + gb

		x = tf.nn.tanh(a) * tf.nn.sigmoid(b)
		#For Skip connection
		s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)

		#For Residual connection
		x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)

		x = (x + residual) * tf.sqrt(0.5)
		return x, s, queue

	def clear_queue(self):
		for conv in [self.conv, self.conv1x1_out, self.conv1x1_skip,
				self.conv1x1c, self.conv1x1g]:
			if conv is not None:
				self.conv.clear_queue()


class ConvTranspose2d:
	def __init__(self, filters, kernel_size, padding, strides):
		freq_axis_kernel_size = kernel_size[0]
		self.convt = tf.layers.Conv2DTranspose(
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding=padding,
			kernel_initializer=tf.constant_initializer(1. / freq_axis_kernel_size, dtype=tf.float32),
			bias_initializer=tf.zeros_initializer(),
			data_format='channels_first')

	def __call__(self, inputs):
		return self.convt(inputs)



def MaskedCrossEntropyLoss(outputs, targets, lengths=None, mask=None, max_len=None):
	if lengths is None and mask is None:
		raise RuntimeError('Please provide either lengths or mask')

	#[batch_size, time_length]
	if mask is None:
		mask = sequence_mask(lengths, max_len, False)

	#One hot encode targets (outputs.shape[-1] = hparams.quantize_channels)
	targets_ = tf.one_hot(targets, depth=tf.shape(outputs)[-1])
	
	with tf.control_dependencies([tf.assert_equal(tf.shape(outputs), tf.shape(targets_))]):
		losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets_)

	with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
		masked_loss = losses * mask

	return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)

def DiscretizedMixtureLogisticLoss(outputs, targets, hparams, lengths=None, mask=None, max_len=None):
	if lengths is None and mask is None:
		raise RuntimeError('Please provide either lengths or mask')

	#[batch_size, time_length, 1]
	if mask is None:
		mask = sequence_mask(lengths, max_len, True)

	#[batch_size, time_length, dimension]
	ones = tf.ones([tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], tf.float32)
	mask_ = mask * ones

	losses = discretized_mix_logistic_loss(
		outputs, targets, num_classes=hparams.quantize_channels,
		log_scale_min=hparams.log_scale_min, reduce=False)

	with tf.control_dependencies([tf.assert_equal(tf.shape(losses), tf.shape(targets))]):
		return tf.reduce_sum(losses * mask_) / tf.reduce_sum(mask_)

def GaussianMaximumLikelihoodEstimation(outputs, targets, hparams, lengths=None, mask=None, max_len=None):
	if lengths is None and mask is None:
		raise RuntimeError('Please provide either lengths or mask')

	#[batch_size, time_length, 1]
	if mask is None:
		mask = sequence_mask(lengths, max_len, True)

	#[batch_size, time_length, dimension]
	ones = tf.ones([tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], tf.float32)
	mask_ = mask * ones

	losses = gaussian_maximum_likelihood_estimation_loss(
		outputs, targets, log_scale_min_gauss=hparams.log_scale_min_gauss, reduce=False)

	with tf.control_dependencies([tf.assert_equal(tf.shape(losses), tf.shape(targets))]):
		return tf.reduce_sum(losses * mask_) / tf.reduce_sum(mask_)