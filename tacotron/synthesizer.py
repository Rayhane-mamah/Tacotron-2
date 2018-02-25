import os
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from utils.text import text_to_sequence
from utils import audio


class Synthesizer:
	def load(self, checkpoint_path, model_name='Tacotron'):
		print('Constructing model: %s' % model_name)
		inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
		input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
		with tf.variable_scope('model') as scope:
			self.model = create_model(model_name, hparams)
			self.model.initialize(inputs, input_lengths)
			self.mel_outputs = self.model.mel_outputs

		print('Loading checkpoint: %s' % checkpoint_path)
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(self.session, checkpoint_path)


	def synthesize(self, text, index ,out_dir):
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seq = text_to_sequence(text, cleaner_names)
		feed_dict = {
			self.model.inputs: [np.asarray(seq, dtype=np.int32)],
			self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
		}
		mels = self.session.run(self.mel_outputs, feed_dict=feed_dict)

		# Write the spectrogram to disk
		mel_filename = 'ljspeech-mel-eval-{:05d}.npy'.format(index)
		np.save(os.path.join(out_dir, mel_filename), mels, allow_pickle=False)

		print('mel spectrograms saved under {}'.format(out_dir))
