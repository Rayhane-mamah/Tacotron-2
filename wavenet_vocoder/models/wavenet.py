import numpy as np 
import tensorflow as tf 

from .modules import Conv1d1x1, ResidualConv1dGLU, ConvTranspose2d, Embedding, ReluActivation
from .mixture import sample_from_discretized_mix_logistic



def _expand_global_features(batch_size, time_length, global_features, data_format='BCT'):
	"""Expand global conditioning features to all time steps

	Args:
		batch_size: int
		time_length: int
		global_features: Tensor of shape [batch_size, channels] or [batch_size, channels, 1]
		data_format: string, 'BCT' to get output of shape [batch_size, channels, time_length]
			or 'BTC' to get output of shape [batch_size, time_length, channels]

	Returns:
		None or Tensor of shape [batch_size, channels, time_length] or [batch_size, time_length, channels]
	"""
	accepted_formats = ['BCT', 'BTC']
	if not (data_format in accepted_formats):
		raise ValueError('{} is an unknow data format, accepted formats are "BCT" and "BTC"'.format(data_format))

	if global_features is None:
		return None

	#[batch_size, channels] ==> [batch_size, channels, 1]
	g = tf.expand_dims(global_features, axis=-1) if tf.rank(global_features) == 2 else global_features
	g_shape = tf.shape(g)

	#[batch_size, channels, 1] ==> [batch_size, channels, time_length]
	ones = tf.ones([g_shape[0], g_shape[1], time_length], tf.float32)
	g = g * ones

	if data_format == 'BCT':
		return g

	else:
		#[batch_size, channels, time_length] ==> [batch_size, time_length, channels]
		return tf.transpose(g, [0, 2, 1])


def receptive_field_size(total_layers, num_cycles, kernel_size, dilation=lambda x: 2**x):
	"""Compute receptive field size.

	Args:
		total_layers; int
		num_cycles: int
		kernel_size: int
		dilation: callable, function used to compute dilation factor.
			use "lambda x: 1" to disable dilated convolutions.

	Returns:
		int: receptive field size in sample.
	"""
	assert total_layers % num_cycles == 0

	layers_per_cycle = total_layers // num_cycles
	dilations = [dilation(i % layers_per_cycle for i in range(total_layers))]
	return (kernel_size - 1) * sum(dilations) + 1


class WaveNet():
	"""Tacotron-2 Wavenet Vocoder model.
	"""
	def __init__(self, hparams):
		#Get hparams
		self._hparams = hparams

		#Initialize model architecture
		assert hparams.layers % hparams.stacks == 0
		layers_per_stack = hparams.layers // hparams.stacks

		#first convolution
		if hparams.scalar_input:
			self.first_conv = Conv1d1x1(1, hparams.residual_channels)
		else:
			self.first_conv = Conv1d1x1(out_channels, hparams.residual_channels)

		#Residual convolutions
		self.conv_layers = [ResidualConv1dGLU(
			hparams.residual_channels, hparams.gate_channels,
			kernel_size=hparams.kernel_size,
			skip_out_channels=hparams.skip_out_channels,
			bias=hparams.use_bias,
			dilation=2**(layer % layers_per_stack), 
			dropout=hparams.dropout,
			cin_channels=hparams.cin_channels,
			gin_channels=hparams.gin_channels,
			weight_normalization=hparams.weight_normalization) for layer in range(hparams.layers)]

		#Final convolutions
		self.last_conv_layers = [
		ReluActivation(name='final_conv_relu1'),
		Conv1d1x1(hparams.skip_out_channels, hparams.skip_out_channels,
			weight_normalization=hparams.weight_normalization),
		ReluActivation(name='final_conv_relu2'),
		Conv1d1x1(hparams.skip_out_channels, hparams.out_channels,
			weight_normalization=hparams.weight_normalization)]

		#Global conditionning embedding
		if hparams.gin_channels > 0 and hparams.use_speaker_embedding:
			assert hparams.n_speakers is not None
			self.embed_speakers = Embedding(
				hparams.n_speakers, hparams.gin_channels, std=0.1, name='gc_embedding')
		else:
			self.embed_speakers = None

		#Upsample conv net
		if hparams.upsample_conditional_features:
			self.upsample_conv = []
			for i, s in enumerate(hparams.upsample_scales):
				freq_axis_padding = (hparams.freq_axis_kernel_size - 1) // 2
				convt = ConvTranspose2d(1, 1, (freq_axis_kernel_size, s),
					padding=(freq_axis_padding, 0),
					dilation=1, stride=(1, s),
					weight_normalization=hparams.weight_normalization)
				self.upsample_conv.append(convt)
				#assuming features are [0, 1] scaled
				#this should avoid negative upsampling output
				self.upsample_conv.append(ReluActivation(name='conditional_upsample_{}'.format(i+1)))
		else:
			self.upsample_conv = None

		self.receptive_field = receptive_field_size(hparams.layers,
			hparams.stacks, hparams.kernel_size)

	#Sanity check functions
	def has_speaker_embedding(self):
		return self.embed_speakers is not None

	def local_conditioning_enabled(self):
		return self.cin_channels > 0

	def step(self, x, c=None, g=None, softmax=False):
		"""Forward step

		Args:
			x: Tensor of shape [batch_size, channels, time_length], One-hot encoded audio signal.
			c: Tensor of shape [batch_size, cin_channels, time_length], Local conditioning features.
			g: Tensor of shape [batch_size, gin_channels, 1] or Ids of shape [batch_size, 1], 
				Global conditioning features.
				Note: set hparams.use_speaker_embedding to False to disable embedding layer and 
				use extrnal One-hot encoded features.
			softmax: Boolean, Whether to apply softmax.

		Returns:
			a Tensor of shape [batch_size, out_channels, time_length]
		"""
		batch_size, _, time_length = tf.shape(x)

		if g is not None:
			if self.embed_speakers is not None:
				#[batch_size, 1] ==> [batch_size, 1, gin_channels]
				g = self.embed_speakers(tf.reshape(g, [batch_size, -1]))
				#[batch_size, gin_channels, 1]
				g = tf.transpose(g, [0, 2, 1])
				assert tf.rank(g) == 3

		#Expand global conditioning features to all time steps
		g_bct = _expand_global_features(batch_size, time_length, g, data_format='BCT')

		if c is not None and self.upsample_conv is not None:
			#[batch_size, 1, cin_channels, time_length]
			c = tf.expand_dims(c, axis=1)
			for transposed_conv in self.upsample_conv:
				c = transposed_conv(c)

			#[batch_size, cin_channels, time_length]
			c = tf.squeeze(c, [1])
			assert c.shape()[-1] == x.shape()[-1]

		#Feed data to network
		x = self.first_conv(x)
		skips = None
		for conv in self.conv_layers:
			x, h = conv(x, c, g_bct)
			if skips is None:
				skips = h
			else:
				skips += h
				skips *= np.sqrt(0.5)
		x = skips

		for conv in self.last_conv_layers:
			x = conv(x)

		return tf.nn.softmax(x, axis=1) if softmax else x

	def _clear_linearized_weights(self):
		self.first_conv._clear_linearized_weight()
		self.last_conv_layers[1]._clear_linearized_weight()
		self.last_conv_layers[-1]._clear_linearized_weight()

	def incremental_step(self, initial_input=None, c=None, g=None,
		time_length=100, test_inputs=None,
		tqdm=lambda x: x, softmax=True, quantize=True,
		log_scale_min=-7.0):
		"""Inceremental forward step

		Inputs of shape [batch_size, channels, time_length] are reshaped to [batch_size, time_length, channels]
		Input of each time step is of shape [batch_size, 1, channels]

		Args:
			Initial input: Tensor of shape [batch_size, channels, 1], initial recurrence input.
			c: Tensor of shape [batch_size, cin_channels, time_length], Local conditioning features
			g: Tensor of shape [batch_size, gin_channels, time_length] or [batch_size, gin_channels, 1]
				global conditioning features
			T: int, number of timesteps to generate
			test_inputs: Tensor, teacher forcing inputs (debug)
			tqdm: callable, tqdm style
			softmax: Boolean, whether to apply softmax activation
			quantize: Whether to quantize softmax output before feeding to
				next time step input
			log_scale_min: float, log scale minimum value.

		Returns:
			Tensor of shape [batch_size, channels, time_length] or [batch_size, channels, 1]
				Generated one_hot encoded samples
		"""
		batch_size = 1

		#Note: should reshape to [batch_size, time_length, channels]
		#not [batch_size, channels, time_length]
		if test_inputs is not None:
			if self._hparams.scalar_input:
				if tf.shape(test_inputs)[1] == 1:
					test_inputs = tf.transpose(test_inputs, [0, 2, 1])
			else:
				if tf.shape(test_inputs)[1] == self._hparams.out_channels:
					test_inputs = tf.transpose(test_inputs, [0, 2, 1])

			batch_size = tf.shape(test_inputs)[0]
			if time_length is None:
				time_length = tf.shape(test_inputs)[1]
			else:
				time_length = max(time_length, tf.shape(test_inputs)[1])

		time_length = int(time_length)

		#Global conditioning
		if g in not None:
			if self.embed_speakers is not None:
				g = self.embed_speakers(tf.reshape(g, [batch_size, -1]))
				#[batch_size, channels, 1]
				g = tf.transpose(g, [0, 2, 1])
				assert tf.rank(g) == 3

		g_btc = _expand_global_features(batch_size, time_length, g, data_format='BTC')

		#Local conditioning
		if c is not None and self.upsample_conv is not None:
			#[batch_size, 1, channels, time_length]
			c = tf.expand_dims(c, axis=1)
			for upsample_conv in self.upsample_conv:
				c = upsample_conv(c)
			#[batch_size, channels, time_length]
			c = tf.squeeze(c, [1])
			assert tf.shape(c)[-1] == time_length

		if c is not None and tf.shape(c)[-1] == time_length:
			c = tf.transpose(c, [0, 2, 1])

		outputs = []
		if initial_input is None:
			if self.scalar_input:
				initial_input = tf.zeros((batch_size, 1 ,1), tf.float32)
			else:
				np_input = np.zeros((batch_size, 1, self._hparams.out_channels))
				np_input[:, :, 127] = 1
				initial_input = tf.convert_to_tensor(np_input)
		else:
			if tf.shape(initial_input)[1] == self._hparams.out_channels:
				initial_input = tf.transpose(initial_input, [0, 2, 1])

		current_input = initial_input
		for t in tqdm(range(time_length)):
			if test_inputs is not None and t < tf.shape(test_inputs)[1]:
				current_input = tf.expand_dims(test_inputs[:, t, :], axis=1)
			else:
				if t > 0:
					current_input = outputs[-1]

			#conditioning features for single time step
			ct = None if c is None else tf.expand_dims(c[:, t, :], axis=1)
			gt = None if g is None else tf.expand_dims(g_btc[:, t, :], axis=1)

			x = current_input
			x = self.first_conv.incremental_step(x)
			skips = None
			for conv in self.conv_layers:
				x, h = conv.incremental_step(x, ct, gt)
				skips = h if skips is None else (skips + h) * np.sqrt(0.5)
			x = skips
			for conv in self.last_conv_layers:
				try:
					x = conv.incremental_step(x)
				except AttributeError:
					x = conv(x)

			#Generate next input by sampling
			if self._hparams.scalar_input:
				x = sample_from_discretized_mix_logistic(
					tf.reshape(x, [batch_size, -1, 1]), log_scale_min=log_scale_min)
			else:
				x = tf.nn.softmax(tf.reshape(x, [batch_size, -1]), axis=1) if softmax \
					else tf.reshape(x, [batch_size, -1])
				if quantize:
					sample = np.random.choice(
						np.arange(self._hparams.out_channels, p=tf.reshape(x, [-1]).eval()))
					new_x = np.zeros(tf.shape(x), np.float32)
					new_x[:, sample] = 1.

					x = tf.convert_to_tensor(new_x, tf.float32)
			outputs.append(x)

		#[time_length, batch_size, channels]
		outputs = tf.stack(outputs)
		#[batch_size, channels, time_length]
		self._clear_linearized_weights()
		return tf.transpose(outputs, [1, 2, 0])
