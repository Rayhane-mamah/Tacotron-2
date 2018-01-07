import tensorflow as tf 
from .zoneout_LSTM import ZoneoutLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple
from hparams import hparams


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
	drop_rate = 0.5

	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d(
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			padding='same')
		batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
		activated = activation(batched)
		return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
								 name='dropout_{}'.format(scope))


def enc_conv_layers(inputs, is_training, kernel_size=(5, ), channels=512, activation=tf.nn.relu, scope=None):
	if scope is None:
		scope = 'enc_conv_layers'

	with tf.variable_scope(scope):
		x = inputs
		for i in range(3):
			x = conv1d(x, kernel_size, channels, activation,
							 	is_training, 'conv_layer_{}_'.format(i + 1)+scope)
	return x

def postnet(inputs, is_training, kernel_size=(5, ), channels=512, activation=tf.nn.tanh, scope=None):
	if scope is None:
		scope = 'dec_conv_layers'

	with tf.variable_scope(scope):
		x = inputs
		for i in range(4):
			x = conv1d(x, kernel_size, channels, activation,
								is_training, 'conv_layer_{}_'.format(i + 1)+scope)
		x = conv1d(x, kernel_size, channels, lambda x:x, is_training, 'conv_layer_{}_'.format(5)+scope)
	return x

def bidirectional_LSTM(inputs, scope, is_training):
	with tf.variable_scope(scope):
		outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
												  ZoneoutLSTMCell(256, 
												  				  is_training, 
												  				  zoneout_factor_cell=0.1,
												  				  zoneout_factor_output=0.1,),
												  ZoneoutLSTMCell(256, 
												  				  is_training,
												  				  zoneout_factor_cell=0.1,
                 												  zoneout_factor_output=0.1,),
												  inputs,
												  dtype=tf.float32)

		#Concatenate c states and h states from forward
		#and backward cells
		encoder_final_state_c = tf.concat(
			(fw_state.c, bw_state.c), 1)
		encoder_final_state_h = tf.concat(
			(fw_state.h, bw_state.h), 1)

		#Get the final state to pass as initial state to decoder
		final_state = LSTMStateTuple(
			c=encoder_final_state_c,
			h=encoder_final_state_h)

	return tf.concat(outputs, axis=2), final_state # Concat forward + backward outputs and final states

def prenet(inputs, is_training, layer_sizes=[128, 128], scope=None):
	x = inputs
	drop_rate = 0.5

	if scope is None:
		scope = 'prenet'

	with tf.variable_scope(scope):
		for i, size in enumerate(layer_sizes):
			dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_{}'.format(i + 1))
			x = tf.layers.dropout(dense, rate=drop_rate, training=is_training,
								  name='dropout_{}_'.format(i + 1) + scope)
	return x


def unidirectional_LSTM(is_training, layers=2, size=512):

	rnn_layers = [ZoneoutLSTMCell(size, is_training, zoneout_factor_cell=0.1,
													 zoneout_factor_output=0.1,
													 ext_proj=hparams.num_mels) for i in range(layers)]

	stacked_LSTM_Cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
	return stacked_LSTM_Cell

def projection(x, shape=512, activation=None, scope=None):
	if scope is None:
		scope = 'linear_projection'

	with tf.variable_scope(scope):
		# if activation==None, this returns a simple linear projection
		# else the projection will be passed through an activation function
		output = tf.contrib.layers.fully_connected(x, shape, activation_fn=activation, 
												   biases_initializer=tf.zeros_initializer(),
												   scope=scope)
		return output

