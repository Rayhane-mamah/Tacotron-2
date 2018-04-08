import numpy as np 
import tensorflow as tf 


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
	"""Simple class to wrap relu activation function in classe for later call.
	"""
	def __init__(self, name=None):
		self.name = name

	def __call__(self, inputs):
		return tf.nn.relu(inputs, name=self.name)


class Conv1d1x1(tf.layers.Conv1D):
	"""Extend tf.layers.Conv1D for dilated layers convolutions.
	"""
	def __init__(in_channels, filters, kernel_size=1, padding='same', dilation_rate=1, use_bias=True, **kwargs):
		super(Conv1d1x1, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			padding=padding,
			dilation_rate=dilation_rate,
			use_bias=use_bias,
			**kwargs)
		self.in_channels = in_channels
		self.input_buffer = None
		self._linearizer_weight = None
		tf.add_to_collections(tf.GraphKeys.UPDATE_OPS, self._clear_linearized_weight)

	def incremental_step(self, inputs):
		#input: [batch_size, time_length, channels]
		if self.training: 
			raise RuntimeError('incremental_step only supports eval mode')


		#reshape weight
		weight = self._get_linearized_weight()
		kw = self.kernel_size[0]
		dilation = self.dilation_rate[0]

		batch_size = tf.shape(inputs)[0]
		if kw > 1:
			if self.input_buffer is None:
				self.input_buffer = tf.zeros((batch_size, kw + (kw - 1) * (dilation - 1), tf.shape(inputs)[2]))
			else:
				#shift buffer
				self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :]
			#append next input
			self.input_buffer[:, -1, :] = inputs[:, -1, :]
			inputs = self.input_buffer
			if dilation > 1:
				inputs = inputs[:, 0::dilation, :]
		output = tf.add(tf.matmul(inputs, weight), self.bias)
		return tf.reshape(output, [batch_size, 1, -1])

	def _get_linearized_weight(self):
		if self._linearizer_weight is None:
			kw = self.kernel.shape[0]
			#layers.Conv1D
			if tf.shape(self.kernel) == (self.filters, self.in_channels, kw):
				weight = tf.transpose(self.kernel, [0, 2, 1])
			else:
				weight = tf.transpose(self.kernel, [2, 0, 1])
			assert tf.shape(weight) == (self.filters, kw, self.in_channels)
			self._linearizer_weight = tf.reshape(self.filters, -1)
		return self._linearizer_weight

	def _clear_linearized_weight(self):
		self._linearizer_weight = None

	def clear_buffer(self):
		self.input_buffer = None

def _conv1x1_forward(conv, x, is_incremental):
	"""conv1x1 step
	"""
	if is_incremental:
		x = conv.incremental_step(x)
	else:
		x = conv(x)

class ResidualConv1dGLU():
	'''Residual dilated conv1d + Gated Linear Unit
	'''

	def __init__(self, residual_channels, gate_channels, kernel_size,
			skip_out_channels=None, cin_channels=-1, gin_channels=-1,
			dropout=1 - .95, padding=None, dilation=1, causal=True,
			bias=True, *args, **kwargs):
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
			padding=padding, dilation=dilation, bias=bias)

		#Local conditioning
		if cin_channels > 0:
			self.conv1x1c = Conv1d1x1(cin_channels, gate_channels,
				bias=bias)
		else:
			self.conv1x1c = None

		#Global conditioning
		if gin_channels > 0:
			self.conv1x1g = Conv1d1x1(gin_channels, gate_channels, 
				bias=bias)
		else:
			self.conv1x1g = None

		gate_out_channels = gate_channels // 2
		self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
		self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, bias=bias)

	def __call__(self, x, c=None, g=None):
		return self.step(x, c, g, False)

	def incremental_step(self, x, c=None, g=None):
		return self.step(x, c, g, True)

	def step(self, x, c, g, is_incremental):
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
			x = self.conv.incremental_step(x)
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

		x = (x + residual) * np.sqrt(0.5)
		return x, s

	def clear_buffer(self):
		for conv in [self.conv, self.conv1x1_out, self.conv1x1_skip,
				self.conv1x1c, self.conv1x1g]:
			if conv is not None:
				self.conv.clear_buffer()