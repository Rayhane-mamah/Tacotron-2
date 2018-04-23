import tensorflow as tf 
from tacotron.models.zoneout_LSTM import ZoneoutLSTMCell
from tensorflow.contrib.rnn import LSTMBlockCell
from hparams import hparams


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
	drop_rate = hparams.tacotron_dropout_rate

	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d(
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			activation=None,
			padding='same')
		batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
		activated = activation(batched)
		return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
								name='dropout_{}'.format(scope))


class EncoderConvolutions:
	"""Encoder convolutional layers used to find local dependencies in inputs characters.
	"""
	def __init__(self, is_training, kernel_size=(5, ), channels=512, activation=tf.nn.relu, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
		super(EncoderConvolutions, self).__init__()
		self.is_training = is_training

		self.kernel_size = kernel_size
		self.channels = channels
		self.activation = activation
		self.scope = 'enc_conv_layers' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i in range(hparams.enc_conv_num_layers):
				x = conv1d(x, self.kernel_size, self.channels, self.activation,
					self.is_training, 'conv_layer_{}_'.format(i + 1)+self.scope)
		return x


class EncoderRNN:
	"""Encoder bidirectional one layer LSTM
	"""
	def __init__(self, is_training, size=256, zoneout=0.1, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control zoneout
			size: integer, the number of LSTM units for each direction
			zoneout: the zoneout factor
			scope: EncoderRNN scope.
		"""
		super(EncoderRNN, self).__init__()
		self.is_training = is_training

		self.size = size
		self.zoneout = zoneout
		self.scope = 'encoder_LSTM' if scope is None else scope

		#Create LSTM Cell
		self._cell = ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout)

	def __call__(self, inputs, input_lengths):
		with tf.variable_scope(self.scope):
			outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
				self._cell,
				self._cell,
				inputs,
				sequence_length=input_lengths,
				dtype=tf.float32)

			return tf.concat(outputs, axis=2) # Concat and return forward + backward outputs


class Prenet:
	"""Two fully connected layers used as an information bottleneck for the attention.
	"""
	def __init__(self, is_training, layer_sizes=[256, 256], activation=tf.nn.relu, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is in training or inference to control dropout
			layer_sizes: list of integers, the length of the list represents the number of pre-net
				layers and the list values represent the layers number of units
			activation: callable, activation functions of the prenet layers.
			scope: Prenet scope.
		"""
		super(Prenet, self).__init__()
		self.drop_rate = hparams.tacotron_dropout_rate

		self.layer_sizes = layer_sizes
		self.is_training = is_training
		self.activation = activation
		
		self.scope = 'prenet' if scope is None else scope

	def __call__(self, inputs):
		x = inputs

		with tf.variable_scope(self.scope):
			for i, size in enumerate(self.layer_sizes):
				dense = tf.layers.dense(x, units=size, activation=self.activation,
					name='dense_{}'.format(i + 1))
				#The paper discussed introducing diversity in generation at inference time
				#by using a dropout of 0.5 only in prenet layers (in both training and inference).
				x = tf.layers.dropout(dense, rate=self.drop_rate, training=True,
					name='dropout_{}'.format(i + 1) + self.scope)
		return x


class DecoderRNN:
	"""Decoder two uni directional LSTM Cells
	"""
	def __init__(self, is_training, layers=2, size=1024, zoneout=0.1, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is in training or inference to control zoneout
			layers: integer, the number of LSTM layers in the decoder
			size: integer, the number of LSTM units in each layer
			zoneout: the zoneout factor
		"""
		super(DecoderRNN, self).__init__()
		self.is_training = is_training

		self.layers = layers
		self.size = size
		self.zoneout = zoneout
		self.scope = 'decoder_rnn' if scope is None else scope

		#Create a set of LSTM layers
		self.rnn_layers = [ZoneoutLSTMCell(size, is_training, 
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout) for i in range(layers)]

		self._cell = tf.contrib.rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=True)

	def __call__(self, inputs, states):
		with tf.variable_scope(self.scope):
			return self._cell(inputs, states)


class FrameProjection:
	"""Projection layer to r * num_mels dimensions or num_mels dimensions
	"""
	def __init__(self, shape=80, activation=None, scope=None):
		"""
		Args:
			shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
			activation: callable, activation function
			scope: FrameProjection scope.
		"""
		super(FrameProjection, self).__init__()

		self.shape = shape
		self.activation = activation
		
		self.scope = 'Linear_projection' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			#If activation==None, this returns a simple Linear projection
			#else the projection will be passed through an activation function
			output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
				name='projection_{}'.format(self.scope))

			return output


class StopProjection:
	"""Projection to a scalar and through a sigmoid activation
	"""
	def __init__(self, is_training, shape=hparams.outputs_per_step, activation=tf.nn.sigmoid, scope=None):
		"""
		Args:
			is_training: Boolean, to control the use of sigmoid function as it is useless to use it
				during training since it is integrate inside the sigmoid_crossentropy loss
			shape: integer, dimensionality of output space. Defaults to 1 (scalar)
			activation: callable, activation function. only used during inference
			scope: StopProjection scope.
		"""
		super(StopProjection, self).__init__()
		self.is_training = is_training
		
		self.shape = shape
		self.activation = activation
		self.scope = 'stop_token_projection' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			output = tf.layers.dense(inputs, units=self.shape,
				activation=None, name='projection_{}'.format(self.scope))

			#During training, don't use activation as it is integrated inside the sigmoid_cross_entropy loss function
			if self.is_training:
				return output
			return self.activation(output)


class Postnet:
	"""Postnet that takes final decoder output and fine tunes it (using vision on past and future frames)
	"""
	def __init__(self, is_training, kernel_size=(5, ), channels=512, activation=tf.nn.tanh, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
		super(Postnet, self).__init__()
		self.is_training = is_training

		self.kernel_size = kernel_size
		self.channels = channels
		self.activation = activation
		self.scope = 'postnet_convolutions' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i in range(hparams.postnet_num_layers - 1):
				x = conv1d(x, self.kernel_size, self.channels, self.activation,
					self.is_training, 'conv_layer_{}_'.format(i + 1)+self.scope)
			x = conv1d(x, self.kernel_size, self.channels, lambda _: _, self.is_training, 'conv_layer_{}_'.format(5)+self.scope)
		return x