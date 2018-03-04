"""Seq2seq layer operations for use in neural networks.
Customized to support dynamic decoding of Tacotron 2.
Only use this dynamic decoder with Tacotron 2. For other applications use the original one from tensorflow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

import tensorflow as tf
from .helpers import TacoTrainingHelper, TacoTestHelper


def _transpose_batch_time(x):
	"""Transpose the batch and time dimensions of a Tensor.
	Retains as much of the static shape information as possible.
	Args:
		x: A tensor of rank 2 or higher.
	Returns:
		x transposed along the first two dimensions.
	Raises:
		ValueError: if `x` is rank 1 or lower.
	"""
	x_static_shape = x.get_shape()
	if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
		raise ValueError(
			"Expected input tensor %s to have rank at least 2, but saw shape: %s" %
			(x, x_static_shape))
	x_rank = array_ops.rank(x)
	x_t = array_ops.transpose(
		x, array_ops.concat(
			([1, 0], math_ops.range(2, x_rank)), axis=0))
	x_t.set_shape(
		tensor_shape.TensorShape([
			x_static_shape[1].value, x_static_shape[0].value
		]).concatenate(x_static_shape[2:]))
	return x_t


def _create_zero_outputs(size, dtype, batch_size):
	"""Create a zero outputs Tensor structure."""
	def _t(s):
		return (s if isinstance(s, ops.Tensor) else constant_op.constant(
			tensor_shape.TensorShape(s).as_list(),
			dtype=dtypes.int32,
			name="zero_suffix_shape"))

	def _create(s, d):
		return array_ops.zeros(
			array_ops.concat(
				([batch_size], _t(s)), axis=0), dtype=d)

	return nest.map_structure(_create, size, dtype)


def dynamic_decode(decoder,
				   output_time_major=False,
				   impute_finished=False,
				   maximum_iterations=None,
				   parallel_iterations=32,
				   swap_memory=False,
				   scope=None):
	"""Perform dynamic decoding with `decoder`.
	Args:
		decoder: A `Decoder` instance.
		output_time_major: Python boolean.  Default: `False` (batch major).  If
			`True`, outputs are returned as time major tensors (this mode is faster).
			Otherwise, outputs are returned as batch major tensors (this adds extra
			time to the computation).
		impute_finished: Python boolean.  If `True`, then states for batch
			entries which are marked as finished get copied through and the
			corresponding outputs get zeroed out.  This causes some slowdown at
			each time step, but ensures that the final state and outputs have
			the correct values and that backprop ignores time steps that were
			marked as finished.
		maximum_iterations: `int32` scalar, maximum allowed number of decoding
			steps.  Default is `None` (decode until the decoder is fully done).
		parallel_iterations: Argument passed to `tf.while_loop`.
		swap_memory: Argument passed to `tf.while_loop`.
		scope: Optional variable scope to use.
	Returns:
		`(final_outputs, final_state)`.
	Raises:
		TypeError: if `decoder` is not an instance of `Decoder`.
		ValueError: if maximum_iterations is provided but is not a scalar.
	"""
	if not isinstance(decoder, tf.contrib.seq2seq.Decoder):
		raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
						type(decoder))

	with variable_scope.variable_scope(scope or "decoder") as varscope:
		# Properly cache variable values inside the while_loop
		if varscope.caching_device is None:
			varscope.set_caching_device(lambda op: op.device)

		if maximum_iterations is not None:
			maximum_iterations = ops.convert_to_tensor(
				maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
			if maximum_iterations.get_shape().ndims != 0:
				raise ValueError("maximum_iterations must be a scalar")

		initial_finished, initial_inputs, initial_state = decoder.initialize()
		zero_outputs = _create_zero_outputs(decoder.output_size,
											decoder.output_dtype,
											decoder.batch_size)

		if maximum_iterations is not None:
			initial_finished = math_ops.logical_or(
				initial_finished, 0 >= maximum_iterations)
		initial_time = constant_op.constant(0, dtype=dtypes.int32)
		stop_error_init = constant_op.constant(0.0, dtype=dtypes.float32)

		def _shape(batch_size, from_shape):
			if not isinstance(from_shape, tensor_shape.TensorShape):
				return tensor_shape.TensorShape(None)
			else:
				batch_size = tensor_util.constant_value(
					ops.convert_to_tensor(
						batch_size, name="batch_size"))
				return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

		def _create_ta(s, d):
			return tensor_array_ops.TensorArray(
				dtype=d,
				size=0,
				dynamic_size=True,
				element_shape=_shape(decoder.batch_size, s))

		initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
												decoder.output_dtype)

		def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
					  finished, unused_error):
			return math_ops.logical_not(math_ops.reduce_all(finished))

		def body(time, outputs_ta, state, inputs, finished, loss):
			"""Internal while_loop body.
			Args:
				time: scalar int32 tensor.
				outputs_ta: structure of TensorArray.
				state: (structure of) state tensors and TensorArrays.
				inputs: (structure of) input tensors.
				finished: 1-D bool tensor.
			Returns:
				`(time + 1, outputs_ta, next_state, next_inputs, next_finished)`.
			"""
			(next_outputs, decoder_state, next_inputs,
			 decoder_finished) = decoder.step(time, inputs, state)

			next_finished = math_ops.logical_or(decoder_finished, finished)
			if maximum_iterations is not None:
				next_finished = math_ops.logical_or(
					next_finished, time + 1 >= maximum_iterations)

			nest.assert_same_structure(state, decoder_state)
			nest.assert_same_structure(outputs_ta, next_outputs)
			nest.assert_same_structure(inputs, next_inputs)

			# Zero out output values past finish
			#next_outputs = tf.reduce_sum(next_outputs, axis=1)
			if impute_finished:
				emit = nest.map_structure(
					lambda out, zero: array_ops.where(finished, zero, out),
					next_outputs,
					zero_outputs)
			else:
				emit = next_outputs

			# Copy through states past finish
			def _maybe_copy_state(new, cur):
				# TensorArrays and scalar states get passed through.
				if isinstance(cur, tensor_array_ops.TensorArray):
					pass_through = True
				else:
					new.set_shape(cur.shape)
					pass_through = (new.shape.ndims == 0)
				return new if pass_through else array_ops.where(finished, cur, new)

			if impute_finished:
				next_state = nest.map_structure(
					_maybe_copy_state, decoder_state, state)
			else:
				next_state = decoder_state

			outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
											outputs_ta, emit)

			#Cumulate <stop_token> loss along decoding steps
			if isinstance(decoder._helper, TacoTrainingHelper):
				stop_token_loss = loss + decoder._helper.stop_token_loss
			elif isinstance(decoder._helper, TacoTestHelper):
				stop_token_loss = loss
			else:
				raise TypeError('Helper used does not belong to any supported Tacotron helpers (TacoTestHelper, TacoTrainingHelper)')

			return (time + 1, outputs_ta, next_state, next_inputs, next_finished, stop_token_loss)

		res = control_flow_ops.while_loop(
			condition,
			body,
			loop_vars=[
				initial_time, initial_outputs_ta, initial_state, initial_inputs,
				initial_finished, stop_error_init
			],
			parallel_iterations=parallel_iterations,
			swap_memory=swap_memory)

		final_outputs_ta = res[1]
		final_state = res[2]

		steps = tf.cast(res[0], tf.float32)
		stop_token_loss = res[5]

		#Average <stop_token> error over decoding steps
		avg_stop_loss = stop_token_loss / steps

		final_outputs = nest.map_structure(
			lambda ta: ta.stack(), final_outputs_ta)
		if not output_time_major:
			final_outputs = nest.map_structure(
				_transpose_batch_time, final_outputs)

	return final_outputs, final_state, avg_stop_loss