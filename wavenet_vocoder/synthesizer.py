import os

import numpy as np
import tensorflow as tf
from datasets.audio import save_wavenet_wav, get_hop_size, melspectrogram
from infolog import log
from wavenet_vocoder.models import create_model
from wavenet_vocoder.train import create_shadow_saver, load_averaged_model
from wavenet_vocoder.feeder import _interp

from . import util


class Synthesizer:
	def load(self, checkpoint_path, hparams, model_name='WaveNet'):
		log('Constructing model: {}'.format(model_name))
		self._hparams = hparams
		local_cond, global_cond = self._check_conditions()

		self.local_conditions = tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='local_condition_features') if local_cond else None
		self.global_conditions = tf.placeholder(tf.int32, shape=(None, 1), name='global_condition_features') if global_cond else None
		self.synthesis_length = tf.placeholder(tf.int32, shape=(), name='synthesis_length') if not local_cond else None
		self.targets = tf.placeholder(tf.float32, shape=(1, None, 1), name='audio_targets') if hparams.wavenet_synth_debug else None #Debug only with 1 wav
		self.input_lengths = tf.placeholder(tf.int32, shape=(1, ), name='input_lengths') if hparams.wavenet_synth_debug else None
		self.synth_debug = hparams.wavenet_synth_debug

		with tf.variable_scope('WaveNet_model') as scope:
			self.model = create_model(model_name, hparams)
			self.model.initialize(y=None, c=self.local_conditions, g=self.global_conditions,
				input_lengths=self.input_lengths, synthesis_length=self.synthesis_length, test_inputs=self.targets)

			self._hparams = hparams
			sh_saver = create_shadow_saver(self.model)

			log('Loading checkpoint: {}'.format(checkpoint_path))
			#Memory allocation on the GPU as needed
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			config.allow_soft_placement = True

			self.session = tf.Session(config=config)
			self.session.run(tf.global_variables_initializer())

		load_averaged_model(self.session, sh_saver, checkpoint_path)

	def synthesize(self, mel_spectrograms, speaker_ids, basenames, out_dir, log_dir):
		hparams = self._hparams
		local_cond, global_cond = self._check_conditions()

		#Switch mels in case of debug
		if self.synth_debug:
			assert len(hparams.wavenet_debug_mels) == len(hparams.wavenet_debug_wavs)
			mel_spectrograms = [np.load(mel_file) for mel_file in hparams.wavenet_debug_mels]

		#Get True length of audio to be synthesized: audio_len = mel_len * hop_size
		audio_lengths = [len(x) * get_hop_size(self._hparams) for x in mel_spectrograms]

		#Prepare local condition batch
		maxlen = max([len(x) for x in mel_spectrograms])
		#[-max, max] or [0,max]
		T2_output_range = (-self._hparams.max_abs_value, self._hparams.max_abs_value) if self._hparams.symmetric_mels else (0, self._hparams.max_abs_value)

		if self._hparams.clip_for_wavenet:
			mel_spectrograms = [np.clip(x, T2_output_range[0], T2_output_range[1]) for x in mel_spectrograms]

		c_batch = np.stack([_pad_inputs(x, maxlen, _pad=T2_output_range[0]) for x in mel_spectrograms]).astype(np.float32)

		if self._hparams.normalize_for_wavenet:
			#rerange to [0, 1]
			c_batch = _interp(c_batch, T2_output_range).astype(np.float32)

		g = None if speaker_ids is None else np.asarray(speaker_ids, dtype=np.int32).reshape(len(c_batch), 1)
		feed_dict = {}

		if local_cond:
			feed_dict[self.local_conditions] = c_batch
		else:
			feed_dict[self.synthesis_length] = 100

		if global_cond:
			feed_dict[self.global_conditions] = g

		if self.synth_debug:
			debug_wavs = hparams.wavenet_debug_wavs
			assert len(debug_wavs) % hparams.wavenet_num_gpus == 0
			test_wavs = [np.load(debug_wav).reshape(-1, 1) for debug_wav in debug_wavs]

			#pad wavs to same length
			max_test_len = max([len(x) for x in test_wavs])
			test_wavs = np.stack([_pad_inputs(x, max_test_len) for x in test_wavs]).astype(np.float32)

			assert len(test_wavs) == len(debug_wavs)
			feed_dict[self.targets] = test_wavs.reshape(len(test_wavs), max_test_len, 1)
			feed_dict[self.input_lengths] = np.asarray([test_wavs.shape[1]])

		#Generate wavs and clip extra padding to select Real speech parts
		generated_wavs, upsampled_features = self.session.run([self.model.tower_y_hat, self.model.tower_synth_upsampled_local_features], feed_dict=feed_dict)

		#Linearize outputs (n_gpus -> 1D)
		generated_wavs = [wav for gpu_wavs in generated_wavs for wav in gpu_wavs]
		upsampled_features = [feat for gpu_feats in upsampled_features for feat in gpu_feats]

		generated_wavs = [generated_wav[:length] for generated_wav, length in zip(generated_wavs, audio_lengths)]
		upsampled_features = [upsampled_feature[:, :length] for upsampled_feature, length in zip(upsampled_features, audio_lengths)]

		audio_filenames = []
		for i, (generated_wav, input_mel, upsampled_feature) in enumerate(zip(generated_wavs, mel_spectrograms, upsampled_features)):
			#Save wav to disk
			audio_filename = os.path.join(out_dir, 'wavenet-audio-{}.wav'.format(basenames[i]))
			save_wavenet_wav(generated_wav, audio_filename, sr=hparams.sample_rate, inv_preemphasize=hparams.preemphasize, k=hparams.preemphasis)
			audio_filenames.append(audio_filename)

			#Compare generated wav mel with original input mel to evaluate wavenet audio reconstruction performance
			#Both mels should match on low frequency information, wavenet mel should contain more high frequency detail when compared to Tacotron mels.
			generated_mel = melspectrogram(generated_wav, hparams).T
			util.plot_spectrogram(generated_mel, os.path.join(log_dir, 'wavenet-mel-spectrogram-{}.png'.format(basenames[i])),
				title='Local Condition vs Reconstructed Audio Mel-Spectrogram analysis', target_spectrogram=input_mel)
			#Save upsampled features to visualize checkerboard artifacts.
			util.plot_spectrogram(upsampled_feature.T, os.path.join(log_dir, 'wavenet-upsampled_features-{}.png'.format(basenames[i])),
				title='Upmsampled Local Condition features', auto_aspect=True)

			#Save waveplot to disk
			if log_dir is not None:
				plot_filename = os.path.join(log_dir, 'wavenet-waveplot-{}.png'.format(basenames[i]))
				util.waveplot(plot_filename, generated_wav, None, hparams, title='WaveNet generated Waveform.')

		return audio_filenames

	def _check_conditions(self):
		local_condition = self._hparams.cin_channels > 0
		global_condition = self._hparams.gin_channels > 0
		return local_condition, global_condition


def _pad_inputs(x, maxlen, _pad=0):
	return np.pad(x, [(0, maxlen - len(x)), (0, 0)], mode='constant', constant_values=_pad)
