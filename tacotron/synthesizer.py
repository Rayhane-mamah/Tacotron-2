import os
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from tacotron.models import create_model
from tacotron.utils.text import text_to_sequence
from tacotron.utils import plot
from datasets import audio
from datetime import datetime


class Synthesizer:
	def load(self, checkpoint_path, gta=False, model_name='Tacotron'):
		print('Constructing model: %s' % model_name)
		inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
		input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
		targets = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'mel_targets')
		with tf.variable_scope('model') as scope:
			self.model = create_model(model_name, hparams)
			if gta:
				self.model.initialize(inputs, input_lengths, targets, gta=gta)
			else:		
				self.model.initialize(inputs, input_lengths)
			self.mel_outputs = self.model.mel_outputs
			self.alignment = self.model.alignments[0]

		self.gta = gta
		print('Loading checkpoint: %s' % checkpoint_path)
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(self.session, checkpoint_path)


	def synthesize(self, text, index, out_dir, log_dir, mel_filename):
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seq = text_to_sequence(text, cleaner_names)
		feed_dict = {
			self.model.inputs: [np.asarray(seq, dtype=np.int32)],
			self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
		}

		if self.gta:
			feed_dict[self.model.mel_targets] = np.load(mel_filename).reshape(1, -1, 80)

		if self.gta or not hparams.predict_linear:
			mels, alignment = self.session.run([self.mel_outputs, self.alignment], feed_dict=feed_dict)

		else:
			linear, mels, alignment = self.session.run([self.linear_outputs, self.mel_outputs, self.alignment], feed_dict=feed_dict)
			linear = linear.reshape(-1, hparams.num_freq)

		mels = mels.reshape(-1, hparams.num_mels) #Thanks to @imdatsolak for pointing this out

		# Write the spectrogram to disk
		# Note: outputs mel-spectrogram files and target ones have same names, just different folders
		mel_filename = os.path.join(out_dir, 'speech-mel-{:05d}.npy'.format(index))
		np.save(mel_filename, mels, allow_pickle=False)

		if log_dir is not None:
			#save wav (mel -> wav)
			wav = audio.inv_mel_spectrogram(mels.T)
			audio.save_wav(wav, os.path.join(log_dir, 'wavs/speech-wav-{:05d}-mel.wav'.format(index)))

			if hparams.predict_linear:
				#save wav (linear -> wav)
				wav = audio.inv_linear_spectrogram(linear.T)
				audio.save_wav(wav, os.path.join(log_dir, 'wavs/speech-wav-{:05d}-linear.wav'.format(index)))

			#save alignments
			plot.plot_alignment(alignment, os.path.join(log_dir, 'plots/speech-alignment-{:05d}.png'.format(index)),
				info='{}'.format(text), split_title=True)

			#save mel spectrogram plot
			plot.plot_spectrogram(mels, os.path.join(log_dir, 'plots/speech-mel-{:05d}.png'.format(index)),
				info='{}'.format(text), split_title=True)

		return mel_filename