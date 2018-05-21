import numpy as np 
import tensorflow as tf 
import os
from infolog import log
from datasets.audio import save_wav
from wavenet_vocoder.models import create_model
from wavenet_vocoder.train import create_shadow_saver, load_averaged_model
from . import util


class Synthesizer:
	def load(self, checkpoint_path, hparams, model_name='WaveNet'):
		log('Constructing model: {}'.format(model_name))
		self._hparams = hparams
		local_cond, global_cond = self._check_conditions()

		self.local_conditions = tf.placeholder(tf.float32, shape=[1, None, hparams.num_mels], name='local_condition_features') if local_cond else None
		self.global_conditions = tf.placeholder(tf.int32, shape=(), name='global_condition_features') if global_cond else None
		self.synthesis_length = tf.placeholder(tf.int32, shape=(), name='synthesis_length') if not local_cond else None

		with tf.variable_scope('model') as scope:
			self.model = create_model(model_name, hparams)
			self.model.initialize(y=None, c=self.local_conditions, g=self.global_conditions,
				input_lengths=None, synthesis_length=self.synthesis_length)

			self._hparams = hparams
			sh_saver = create_shadow_saver(self.model)

			log('Loading checkpoint: {}'.format(checkpoint_path))
			self.session = tf.Session()
			self.session.run(tf.global_variables_initializer())
			load_averaged_model(self.session, sh_saver, checkpoint_path)

	def synthesize(self, mel_spectrogram, speaker_id, index, out_dir, log_dir):
		hparams = self._hparams
		local_cond, global_cond = self._check_conditions()

		c = mel_spectrogram
		g = speaker_id
		feed_dict = {}

		if local_cond:
			feed_dict[self.local_conditions] = [np.array(c, dtype=np.float32)]
		else:
			feed_dict[self.synthesis_length] = 100

		if global_cond:
			feed_dict[self.global_conditions] = [np.array(g, dtype=np.int32)]

		generated_wav = self.session.run(self.model.y_hat, feed_dict=feed_dict)

		#Save wav to disk
		audio_filename = os.path.join(out_dir, 'speech-audio-{:05d}.wav'.format(index))
		save_wav(generated_wav, audio_filename, sr=hparams.sample_rate)

		#Save waveplot to disk
		if log_dir is not None:
			plot_filename = os.path.join(log_dir, 'speech-waveplot-{:05d}.png'.format(index))
			util.waveplot(plot_filename, generated_wav, None, hparams)

		return audio_filename

	def _check_conditions(self):
		local_condition = self._hparams.cin_channels > 0
		global_condition = self._hparams.gin_channels > 0
		return local_condition, global_condition
