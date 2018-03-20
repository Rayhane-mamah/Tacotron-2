import numpy as np 
import os
import threading
import time
import traceback
from utils.text import text_to_sequence
from utils.infolog import log
import tensorflow as tf 


_batches_per_group = 32
_pad = 0
_token_pad = 1.

class Feeder(threading.Thread):
	"""
		Feeds batches of data into queue on a background thread.
	"""

	def __init__(self, coordinator, metadata_filename, hparams):
		super(Feeder, self).__init__()
		self._coord = coordinator
		self._hparams = hparams
		self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		self._offset = 0

		# Load metadata
		self._datadir = os.path.dirname(metadata_filename)
		with open(metadata_filename, encoding='utf-8') as f:
			self._metadata = [line.strip().split('|') for line in f]
			frame_shift_ms = hparams.hop_size / hparams.sample_rate
			hours = sum([int(x[1]) for x in self._metadata]) * frame_shift_ms / (3600)
			log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(self._metadata), hours))

		# Create placeholders for inputs and targets. Don't specify batch size because we want
		# to be able to feed different batch sizes at eval time.
		self._placeholders = [
		tf.placeholder(tf.int32, shape=(None, None), name='inputs'),
		tf.placeholder(tf.int32, shape=(None, ), name='input_lengths'),
		tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='mel_targets'),
		tf.placeholder(tf.float32, shape=(None, None), name='token_targets'),
		]

		# Create queue for buffering data
		queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32], name='input_queue')
		self._enqueue_op = queue.enqueue(self._placeholders)
		self.inputs, self.input_lengths, self.mel_targets, self.token_targets = queue.dequeue()
		self.inputs.set_shape(self._placeholders[0].shape)
		self.input_lengths.set_shape(self._placeholders[1].shape)
		self.mel_targets.set_shape(self._placeholders[2].shape)
		self.token_targets.set_shape(self._placeholders[3].shape)

	def start_in_session(self, session):
		self._session = session
		self.start()

	def run(self):
		try:
			while not self._coord.should_stop():
				self._enqueue_next_group()
		except Exception as e:
			traceback.print_exc()
			self._coord.request_stop(e)

	def _enqueue_next_group(self):
		start = time.time()

		# Read a group of examples
		n = self._hparams.batch_size
		r = self._hparams.outputs_per_step
		examples = [self._get_next_example() for i in range(n * _batches_per_group)]

		# Bucket examples based on similar output sequence length for efficiency
		examples.sort(key=lambda x: x[-1])
		batches = [examples[i: i+n] for i in range(0, len(examples), n)]
		np.random.shuffle(batches)

		log('\nGenerated {} batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
		for batch in batches:
			feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
			self._session.run(self._enqueue_op, feed_dict=feed_dict)

	def _get_next_example(self):
		"""
		Gets a single example (input, mel_target, cost) from disk
		"""
		if self._offset >= len(self._metadata):
			self._offset = 0
			np.random.shuffle(self._metadata)
		meta = self._metadata[self._offset]
		self._offset += 1

		text = meta[2]

		input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
		mel_target = np.load(os.path.join(self._datadir, meta[0]))
		token_target = np.asarray([0.] * len(mel_target))
		return (input_data, mel_target, token_target, len(mel_target))


def _prepare_batch(batch, outputs_per_step):
	np.random.shuffle(batch)
	inputs = _prepare_inputs([x[0] for x in batch])
	input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
	mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
	token_targets = _prepare_token_targets([x[2] for x in batch], outputs_per_step)
	return (inputs, input_lengths, mel_targets, token_targets)

def _prepare_inputs(inputs):
	max_len = max([len(x) for x in inputs])
	return np.stack([_pad_input(x, max_len) for x in inputs])

def _prepare_targets(targets, alignment):
	max_len = max([len(t) for t in targets]) + 1 
	return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])

def _prepare_token_targets(targets, alignment):
	max_len = max([len(t) for t in targets]) + 1
	return np.stack([_pad_token_target(t, _round_up(max_len, alignment)) for t in targets])

def _pad_input(x, length):
	return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _pad_target(t, length):
	return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=_pad)

def _pad_token_target(t, length):
	return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=_token_pad)

def _round_up(x, multiple):
	remainder = x % multiple
	return x if remainder == 0 else x + multiple - remainder
