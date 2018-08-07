import argparse
import os

import numpy as np
import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tqdm import tqdm
from wavenet_vocoder.synthesizer import Synthesizer


def run_synthesis(args, checkpoint_path, output_dir, hparams):
	log_dir = os.path.join(output_dir, 'plots')
	wav_dir = os.path.join(output_dir, 'wavs')

	#We suppose user will provide correct folder depending on training method
	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	if args.model in ('Both', 'Tacotron-2'):
		#If running all Tacotron-2, synthesize audio from evaluated mels
		metadata_filename = os.path.join(args.mels_dir, 'map.txt')
		with open(metadata_filename, encoding='utf-8') as f:
			metadata = [line.strip().split('|') for line in f]
			frame_shift_ms = hparams.hop_size / hparams.sample_rate
			hours = sum([int(x[-1]) for x in metadata]) * frame_shift_ms / (3600)
			log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

		metadata = np.array(metadata)
		speaker_ids = metadata[:, 2]
		mel_files = metadata[:, 1]
		texts = metadata[:, 0]
	else:
		#else Get all npy files in input_dir (supposing they are mels)
		mel_files  = [os.path.join(args.mels_dir, f) for f in os.listdir(args.mels_dir) if f.split('.')[-1] == 'npy']
		speaker_ids = args.speaker_id
		texts = None

	log('Starting synthesis! (this will take a while..)')
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)

	with open(os.path.join(wav_dir, 'map.txt'), 'w') as file:
		for i, mel_file in enumerate(tqdm(mel_files)):
			mel_spectro = np.load(mel_file)
			if hparams.normalize_for_wavenet:
				#[-max, max] or [0,max]
				T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)
				#rerange to [0, 1]
				mel_spectro = np.interp(mel_spectro, T2_output_range, (0, 1))

			basename = mel_file.replace('.npy', '')
			speaker_id = speaker_ids[i]
			audio_file = synth.synthesize(mel_spectro, speaker_id, basename, wav_dir, log_dir)

			if texts is None:
				file.write('{}|{}\n'.format(mel_file, audio_file))
			else:
				file.write('{}|{}|{}\n'.format(texts[i], mel_file, audio_file))

	log('synthesized audio waveforms at {}'.format(wav_dir))



def wavenet_synthesize(args, hparams, checkpoint):
	output_dir = 'wavenet_' + args.output_dir

	try:
		checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
		log('loaded model at {}'.format(checkpoint_path))
	except AttributeError:
		#Swap logs dir name in case user used Tacotron-2 for train and Both for test (and vice versa)
		if 'Both' in checkpoint:
			checkpoint = checkpoint.replace('Both', 'Tacotron-2')
		elif 'Tacotron-2' in checkpoint:
			checkpoint = checkpoint.replace('Tacotron-2', 'Both')
		else: #Synthesizing separately
			raise AssertionError('Cannot restore checkpoint: {}, did you train a model?'.format(checkpoint))

		try:
			#Try loading again
			checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
			log('loaded model at {}'.format(checkpoint_path))
		except:
			raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

	run_synthesis(args, checkpoint_path, output_dir, hparams)
