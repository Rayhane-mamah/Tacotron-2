import tensorflow as tf 
from .zoneout_LSTM import ZoneoutLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple
from hparams import hparams


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
	drop_rate = hparams.dropout_rate

	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d(
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			activation=activation,
			padding='same')
		batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
		return tf.layers.dropout(batched, rate=drop_rate, training=is_training,
		 						name='dropout_{}'.format(scope))

def enc_conv_layers(inputs, is_training, kernel_size=(5, ), channels=512, activation=tf.nn.relu, scope=None):
	if scope is None:
		scope = 'enc_conv_layers'

	with tf.variable_scope(scope):
		x = inputs
		for i in range(hparams.enc_conv_num_layers):
			x = conv1d(x, kernel_size, channels, activation,
							 	is_training, 'conv_layer_{}_'.format(i + 1)+scope)
	return x

def postnet(inputs, is_training, kernel_size=(5, ), channels=512, activation=tf.nn.tanh, scope=None):
	if scope is None:
		scope = 'dec_conv_layers'

	with tf.variable_scope(scope):
		x = inputs
		for i in range(hparams.postnet_num_layers - 1):
			x = conv1d(x, kernel_size, channels, activation,
								is_training, 'conv_layer_{}_'.format(i + 1)+scope)
		x = conv1d(x, kernel_size, channels, lambda _: _, is_training, 'conv_layer_{}_'.format(5)+scope)
	return x

def bidirectional_LSTM(inputs, input_lengths, scope, is_training, size=256, zoneout=0.1):
	with tf.variable_scope(scope):
		outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
												  ZoneoutLSTMCell(size, 
												  				  is_training, 
												  				  zoneout_factor_cell=zoneout,
												  				  zoneout_factor_output=zoneout),
												  ZoneoutLSTMCell(size, 
												  				  is_training,
												  				  zoneout_factor_cell=zoneout,
                 												  zoneout_factor_output=zoneout),
												  inputs,
												  sequence_length=input_lengths,
												  dtype=tf.float32)

		#Concatenate c states and h states from forward
		#and backward cells
		encoder_final_state_c = tf.concat(
			(fw_state.c, bw_state.c), 1)
		encoder_final_state_h = tf.concat(
			(fw_state.h, bw_state.h), 1)

		#Get the final state, we don't really use it in our case
		#I'll keep it just in case
		final_state = LSTMStateTuple(
			c=encoder_final_state_c,
			h=encoder_final_state_h)

		return tf.concat(outputs, axis=2), final_state # Concat forward + backward outputs and return with final states

def unidirectional_LSTM(input_cell, is_training, layers=2, size=512, zoneout=0.1):
	#Create a set of LSTM layers
	rnn_layers = [ZoneoutLSTMCell(size, is_training, 
								  zoneout_factor_cell=zoneout,
								  zoneout_factor_output=zoneout) for i in range(layers)]

	#Add the first concatenation layer wrapper
	rnn_layers = [input_cell] + rnn_layers

	return tf.nn.rnn_cell.MultiRNNCell(rnn_layers, state_is_tuple=True)

def prenet(inputs, is_training, layer_sizes=[128, 128], scope=None):
	x = inputs
	drop_rate = hparams.dropout_rate

	if scope is None:
		scope = 'prenet'

	with tf.variable_scope(scope):
		for i, size in enumerate(layer_sizes):
			dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_{}'.format(i + 1))
			#The paper discussed introducing diversity in generation at inference time
			#by using a dropout of 0.5 only in prenet layers.
			x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, 
								  name='dropout_{}_'.format(i + 1) + scope)
	return x

def projection(x, shape=80, activation=None, scope=None):
	if scope is None:
		scope = 'linear_projection'

	with tf.variable_scope(scope):
		# if activation==None, this returns a simple linear projection
		# else the projection will be passed through an activation function
		output = tf.contrib.layers.fully_connected(x, shape, activation_fn=activation, 
												   biases_initializer=tf.zeros_initializer(),
												   scope=scope)
		return output

def stop_token_projection(x, shape=1, activation=lambda _: _, weights_name='stop_token_weights', bias_name='step_token_bias'):
	"""Just making sure we use the same weights for training and
	inference time for stop token prediction
	"""

	st_W = tf.get_variable(weights_name, shape=[x.shape[-1], 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
	st_b = tf.get_variable(bias_name, shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())

	output = activation(tf.add(tf.matmul(x, st_W), st_b))

	return output