"""Attention file for location based attention (compatible with tensorflow attention wrapper)"""

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism
from tensorflow.python.ops import nn_ops
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from hparams import hparams




def _location_sensitive_score(W_query, attention_weights, W_keys):
	"""Impelements Bahdanau-style (cumulative) scoring function.
	This attention is described in:
	J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
  gio, “Attention-based models for speech recognition,” in Ad-
  vances in Neural Information Processing Systems, 2015, pp.
  577–585.

  #######################################################################
            hybrid attention (content-based + location-based)
        				     f = F * α_{i-1}
     energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f)))
  #######################################################################

  Args:
	W_query: Tensor, shape '[batch_size, num_units]' to compare to location features.
	attention_weights (alignments): previous attention weights, shape '[batch_size, max_time]'
  Returns:
	A '[batch_size, max_time]'
	"""
	dtype = W_query.dtype
	# Get the number of hidden units from the trailing dimension of query
	num_units = W_query.shape[-1].value or array_ops.shape(W_query)[-1]

	# [batch_size, max_time] -> [batch_size, max_time, 1]
	attention_weights = tf.expand_dims(attention_weights, axis=2)
	# location features [batch_size, max_time, filters]
	f = tf.layers.conv1d(attention_weights, filters=hparams.attention_filters,
		kernel_size=hparams.attention_kernel, padding='same',
		kernel_initializer=tf.contrib.layers.xavier_initializer(),
		use_bias=False,
		name='location_features')

	# Projected location features [batch_size, max_time, attention_dim]
	W_fil = tf.contrib.layers.fully_connected(
		f,
		num_outputs=num_units,
		activation_fn=None,
		weights_initializer=tf.contrib.layers.xavier_initializer(),
		biases_initializer=None,
		scope='W_filter')

	v_a = tf.get_variable(
		'v_a', shape=[num_units], dtype=tf.float32,
		initializer=tf.contrib.layers.xavier_initializer())

	return tf.reduce_sum(v_a * tf.tanh(W_keys + tf.expand_dims(W_query, axis=1) + W_fil), axis=2)


class LocationSensitiveAttention(_BaseAttentionMechanism):
	"""Impelements Bahdanau-style (cumulative) scoring function.
	Usually referred to as "hybrid" attention (content-based + location-based)
	This attention is described in:
	J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
  gio, “Attention-based models for speech recognition,” in Ad-
  vances in Neural Information Processing Systems, 2015, pp.
  577–585.
	"""

	def __init__(self,
				 num_units,
				 memory,
				 memory_sequence_length=None,
				 probability_fn=None,
				 score_mask_value=tf.float32.min,
				 name='LocationSensitiveAttention'):
		"""Construct the Attention mechanism.
		Args:
			num_units: The depth of the query mechanism.
			memory: The memory to query; usually the output of an RNN encoder.  This
				tensor should be shaped `[batch_size, max_time, ...]`.
			memory_sequence_length (optional): Sequence lengths for the batch entries
				in memory.  If provided, the memory tensor rows are masked with zeros
				for values past the respective sequence lengths.
			probability_fn: (optional) A `callable`.  Converts the score to
				probabilities.  The default is @{tf.nn.softmax}. Other options include
				@{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
				Its signature should be: `probabilities = probability_fn(score)`.
			score_mask_value: (optional): The mask value for score before passing into
				`probability_fn`. The default is -inf. Only used if
				`memory_sequence_length` is not None.
			name: Name to use when creating ops.
		"""
		if probability_fn is None:
			probability_fn = nn_ops.softmax
		wrapped_probability_fn = lambda score, _: probability_fn(score)
		super(LocationSensitiveAttention, self).__init__(
				query_layer=layers_core.Dense(
						num_units, name='query_layer', use_bias=False),
				memory_layer=layers_core.Dense(
						num_units, name='memory_layer', use_bias=False),
				memory=memory,
				probability_fn=wrapped_probability_fn,
				memory_sequence_length=memory_sequence_length,
				score_mask_value=score_mask_value,
				name=name)
		self._num_units = num_units
		self._name = name

	def get_alignments(self, query, previous_alignments):
		"""Score the query based on the keys and values.
		Args:
			query: Tensor of dtype matching `self.values` and shape
				`[batch_size, query_depth]`.
			previous_alignments: Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]`
				(`alignments_size` is memory's `max_time`).
		Returns:
			alignments: Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]` (`alignments_size` is memory's
				`max_time`).
		"""
		with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):
			# processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
			processed_query = self.query_layer(query) if self.query_layer else query
			# energy shape [batch_size, max_time]
			energy = _location_sensitive_score(processed_query, previous_alignments, self._keys)
		# alignments shape = energy shape = [batch_size, max_time]
		alignments = self._probability_fn(energy, previous_alignments)
		return alignments


	def __call__(self, query_vector, previous_alignments):
		"""Computes the context vector and alignments.
		"""
		alignments = self.get_alignments(query_vector, previous_alignments)

		# Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
		expanded_alignments = array_ops.expand_dims(alignments, 1)

		# Context is the inner product of alignments and values along the
		# memory time dimension.
		# alignments shape is
		#   [batch_size, 1, memory_time]
		# attention_mechanism.values shape is
		#   [batch_size, memory_time, memory_size]
		# the batched matmul is over memory_time, so the output shape is
		#   [batch_size, 1, memory_size].
		# we then squeeze out the singleton dim.
		context = math_ops.matmul(expanded_alignments, self.values)
		context = array_ops.squeeze(context, [1])

		return context, alignments