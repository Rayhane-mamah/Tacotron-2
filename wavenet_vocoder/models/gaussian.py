import numpy as np
import tensorflow as tf


def gaussian_maximum_likelihood_estimation_loss(y_hat, y, log_scale_min_gauss, num_classes, use_cdf=True, reduce=True):
	'''compute the gaussian MLE loss'''
	with tf.control_dependencies([tf.assert_equal(tf.shape(y_hat)[1], 2), tf.assert_equal(tf.rank(y_hat), 3)]):
		#[batch_size, time_steps, channels]
		y_hat = tf.transpose(y_hat, [0, 2, 1])

	#Unpack parameters: mean and log_scale outputs
	mean = y_hat[:, :, 0]
	log_scale = tf.maximum(y_hat[:, :, 1], log_scale_min_gauss)
	y = tf.squeeze(y, [-1])

	if use_cdf:
		#Compute log_probs using CDF trick (Normalized loss value and more stable training than with natural log prob)
		#Instantiate a Normal distribution with model outputs
		gaussian = tf.contrib.distributions.Normal(loc=mean, scale=tf.exp(log_scale))

		#Draw CDF+ and CDF- neighbors to the true sample y
		cdf_plus = gaussian.cdf(y + 1. / (num_classes - 1))
		cdf_min = gaussian.cdf(y - 1. / (num_classes - 1))

		#Maximize the difference between CDF+ and CDF- (or its log)
		log_prob = tf.log(tf.maximum(cdf_plus - cdf_min, 1e-12))

	else:
		#Get log probability of each sample under this distribution in a computationally stable fashion
		#This is the log(PDF)
		log_prob = -0.5 * (np.log(2. * np.pi) + 2. * log_scale + tf.square(y - mean) * tf.exp(-2. * log_scale))

	#Loss (Maximize log probability by minimizing its negative)
	if reduce:
		return -tf.reduce_sum(log_prob)
	else:
		return -tf.expand_dims(log_prob, [-1])

def sample_from_gaussian(y, log_scale_min_gauss):
	'''sample from learned gaussian distribution'''
	with tf.control_dependencies([tf.assert_equal(tf.shape(y)[1], 2)]):
		#[batch_size, time_length, channels]
		y = tf.transpose(y, [0, 2, 1])

	mean = y[:, :, 0]
	log_scale = tf.maximum(y[:, :, 1], log_scale_min_gauss)
	scale = tf.exp(log_scale)

	gaussian_dist = tf.contrib.distributions.Normal(loc=mean, scale=scale, allow_nan_stats=False)
	x = gaussian_dist.sample()

	return tf.minimum(tf.maximum(x, -1.), 1.)
