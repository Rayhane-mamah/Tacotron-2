import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import infolog
import librosa
import numpy as np
import tensorflow as tf
from hparams import hparams_debug_string
from datasets.audio import save_wavenet_wav, melspectrogram
from tacotron.utils import ValueWindow
from wavenet_vocoder.feeder import Feeder, _interp
from wavenet_vocoder.models import create_model

from . import util

log = infolog.log


def time_string():
	return datetime.now().strftime('%Y-%m-%d %H:%M')

def add_embedding_stats(summary_writer, embedding_names, paths_to_meta, checkpoint_path):
	#Create tensorboard projector
	config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
	config.model_checkpoint_path = checkpoint_path

	for embedding_name, path_to_meta in zip(embedding_names, paths_to_meta):
		#Initialize config
		embedding = config.embeddings.add()
		#Specifiy the embedding variable and the metadata
		embedding.tensor_name = embedding_name
		embedding.metadata_path = path_to_meta
	
	#Project the embeddings to space dimensions for visualization
	tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)

def add_train_stats(model, hparams):
	with tf.variable_scope('stats') as scope:
		for i in range(hparams.wavenet_num_gpus):
			tf.summary.histogram('wav_outputs %d' % i, model.tower_y_hat_log[i])
			tf.summary.histogram('wav_targets %d' % i, model.tower_y_log[i])
			if model.tower_means[i] is not None:
				tf.summary.histogram('gaussian_means %d' % i, model.tower_means[i])
				tf.summary.histogram('gaussian_log_scales %d' % i, model.tower_log_scales[i])

		tf.summary.scalar('wavenet_learning_rate', model.learning_rate)
		tf.summary.scalar('wavenet_loss', model.loss)

		gradient_norms = [tf.norm(grad) for grad in model.gradients if grad is not None]
		tf.summary.histogram('gradient_norm', gradient_norms)
		tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)) #visualize gradients (in case of explosion)
		return tf.summary.merge_all()

def add_test_stats(summary_writer, step, eval_loss, hparams):
	values = [
	tf.Summary.Value(tag='Wavenet_eval_model/eval_stats/wavenet_eval_loss', simple_value=eval_loss),
	]

	test_summary = tf.Summary(value=values)
	summary_writer.add_summary(test_summary, step)


def create_shadow_saver(model, global_step=None):
	'''Load shadow variables of saved model.

	Inspired by: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

	Can also use: shadow_dict = model.ema.variables_to_restore()
	'''
	#Add global step to saved variables to save checkpoints correctly
	shadow_variables = [model.ema.average_name(v) for v in model.variables]
	variables = model.variables

	if global_step is not None:
		shadow_variables += ['global_step']
		variables += [global_step]

	shadow_dict = dict(zip(shadow_variables, variables)) #dict(zip(keys, values)) -> {key1: value1, key2: value2, ...}
	return tf.train.Saver(shadow_dict, max_to_keep=20)

def load_averaged_model(sess, sh_saver, checkpoint_path):
	sh_saver.restore(sess, checkpoint_path)


def eval_step(sess, global_step, model, plot_dir, wav_dir, summary_writer, hparams, model_name):
	'''Evaluate model during training.
	Supposes that model variables are averaged.
	'''
	start_time = time.time()
	y_hat, y_target, loss, input_mel, upsampled_features = sess.run([model.tower_y_hat[0], model.tower_y_target[0],
		model.eval_loss, model.tower_eval_c[0], model.tower_eval_upsampled_local_features[0]])
	duration = time.time() - start_time
	log('Time Evaluation: Generation of {} audio frames took {:.3f} sec ({:.3f} frames/sec)'.format(
		len(y_target), duration, len(y_target)/duration))

	#Make audio and plot paths
	pred_wav_path = os.path.join(wav_dir, 'step-{}-pred.wav'.format(global_step))
	target_wav_path = os.path.join(wav_dir, 'step-{}-real.wav'.format(global_step))
	plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(global_step))
	mel_path = os.path.join(plot_dir, 'step-{}-reconstruction-mel-spectrogram.png'.format(global_step))
	upsampled_path = os.path.join(plot_dir, 'step-{}-upsampled-features.png'.format(global_step))

	#Save figure
	util.waveplot(plot_path, y_hat, y_target, model._hparams, title='{}, {}, step={}, loss={:.5f}'.format(model_name, time_string(), global_step, loss))
	log('Eval loss for global step {}: {:.3f}'.format(global_step, loss))

	#Compare generated wav mel with original input mel to evaluate wavenet audio reconstruction performance
	#Both mels should match on low frequency information, wavenet mel should contain more high frequency detail when compared to Tacotron mels.
	T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)
	generated_mel = _interp(melspectrogram(y_hat, hparams).T, T2_output_range)
	util.plot_spectrogram(generated_mel, mel_path, title='Local Condition vs Reconst. Mel-Spectrogram, step={}, loss={:.5f}'.format(
		global_step, loss), target_spectrogram=input_mel.T)
	util.plot_spectrogram(upsampled_features.T, upsampled_path, title='Upsampled Local Condition features, step={}, loss={:.5f}'.format(
		global_step, loss), auto_aspect=True)

	#Save Audio
	save_wavenet_wav(y_hat, pred_wav_path, sr=hparams.sample_rate, inv_preemphasize=hparams.preemphasize, k=hparams.preemphasis)
	save_wavenet_wav(y_target, target_wav_path, sr=hparams.sample_rate, inv_preemphasize=hparams.preemphasize, k=hparams.preemphasis)

	#Write eval summary to tensorboard
	log('Writing eval summary!')
	add_test_stats(summary_writer, global_step, loss, hparams=hparams)

def save_log(sess, global_step, model, plot_dir, wav_dir, hparams, model_name):
	log('\nSaving intermediate states at step {}'.format(global_step))
	idx = 0
	y_hat, y, loss, length, input_mel, upsampled_features = sess.run([model.tower_y_hat_log[0][idx], 
		model.tower_y_log[0][idx], 
		model.loss,
		model.tower_input_lengths[0][idx], 
		model.tower_c[0][idx], model.tower_upsampled_local_features[0][idx]])

	#mask by length
	y_hat[length:] = 0
	y[length:] = 0

	#Make audio and plot paths
	pred_wav_path = os.path.join(wav_dir, 'step-{}-pred.wav'.format(global_step))
	target_wav_path = os.path.join(wav_dir, 'step-{}-real.wav'.format(global_step))
	plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(global_step))
	mel_path = os.path.join(plot_dir, 'step-{}-reconstruction-mel-spectrogram.png'.format(global_step))
	upsampled_path = os.path.join(plot_dir, 'step-{}-upsampled-features.png'.format(global_step))

	#Save figure
	util.waveplot(plot_path, y_hat, y, hparams, title='{}, {}, step={}, loss={:.5f}'.format(model_name, time_string(), global_step, loss))

	#Compare generated wav mel with original input mel to evaluate wavenet audio reconstruction performance
	#Both mels should match on low frequency information, wavenet mel should contain more high frequency detail when compared to Tacotron mels.
	T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)
	generated_mel = _interp(melspectrogram(y_hat, hparams).T, T2_output_range)
	util.plot_spectrogram(generated_mel, mel_path, title='Local Condition vs Reconst. Mel-Spectrogram, step={}, loss={:.5f}'.format(
		global_step, loss), target_spectrogram=input_mel.T)
	util.plot_spectrogram(upsampled_features.T, upsampled_path, title='Upsampled Local Condition features, step={}, loss={:.5f}'.format(
		global_step, loss), auto_aspect=True)

	#Save audio
	save_wavenet_wav(y_hat, pred_wav_path, sr=hparams.sample_rate, inv_preemphasize=hparams.preemphasize, k=hparams.preemphasis)
	save_wavenet_wav(y, target_wav_path, sr=hparams.sample_rate, inv_preemphasize=hparams.preemphasize, k=hparams.preemphasis)

def save_checkpoint(sess, saver, checkpoint_path, global_step):
	saver.save(sess, checkpoint_path, global_step=global_step)


def model_train_mode(args, feeder, hparams, global_step, init=False):
	with tf.variable_scope('WaveNet_model', reuse=tf.AUTO_REUSE) as scope:
		model_name = None
		if args.model == 'Tacotron-2':
			model_name = 'WaveNet'
		model = create_model(model_name or args.model, hparams, init)
		#initialize model to train mode
		model.initialize(feeder.targets, feeder.local_condition_features, feeder.global_condition_features,
			feeder.input_lengths, x=feeder.inputs)
		model.add_loss()
		model.add_optimizer(global_step)
		stats = add_train_stats(model, hparams)
		return model, stats

def model_test_mode(args, feeder, hparams, global_step):
	with tf.variable_scope('WaveNet_model', reuse=tf.AUTO_REUSE) as scope:
		model_name = None
		if args.model == 'Tacotron-2':
			model_name = 'WaveNet'
		model = create_model(model_name or args.model, hparams)
		#initialize model to test mode
		model.initialize(feeder.eval_targets, feeder.eval_local_condition_features, feeder.eval_global_condition_features,
			feeder.eval_input_lengths)
		model.add_loss()
		return model

def train(log_dir, args, hparams, input_path):
	save_dir = os.path.join(log_dir, 'wave_pretrained')
	plot_dir = os.path.join(log_dir, 'plots')
	wav_dir = os.path.join(log_dir, 'wavs')
	eval_dir = os.path.join(log_dir, 'eval-dir')
	eval_plot_dir = os.path.join(eval_dir, 'plots')
	eval_wav_dir = os.path.join(eval_dir, 'wavs')
	tensorboard_dir = os.path.join(log_dir, 'wavenet_events')
	meta_folder = os.path.join(log_dir, 'metas')
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(plot_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(eval_plot_dir, exist_ok=True)
	os.makedirs(eval_wav_dir, exist_ok=True)
	os.makedirs(tensorboard_dir, exist_ok=True)
	os.makedirs(meta_folder, exist_ok=True)

	checkpoint_path = os.path.join(save_dir, 'wavenet_model.ckpt')
	input_path = os.path.join(args.base_dir, input_path)

	log('Checkpoint_path: {}'.format(checkpoint_path))
	log('Loading training data from: {}'.format(input_path))
	log('Using model: {}'.format(args.model))
	log(hparams_debug_string())

	#Start by setting a seed for repeatability
	tf.set_random_seed(hparams.wavenet_random_seed)

	#Set up data feeder
	coord = tf.train.Coordinator()
	with tf.variable_scope('datafeeder') as scope:
		feeder = Feeder(coord, input_path, args.base_dir, hparams)

	#Set up model
	global_step = tf.Variable(0, name='global_step', trainable=False)
	model, stats = model_train_mode(args, feeder, hparams, global_step)
	eval_model = model_test_mode(args, feeder, hparams, global_step)

	#Speaker Embeddings metadata
	if hparams.speakers_path is not None:
		speaker_embedding_meta = hparams.speakers_path

	else:
		speaker_embedding_meta = os.path.join(meta_folder, 'SpeakerEmbeddings.tsv')
		if not os.path.isfile(speaker_embedding_meta):
			with open(speaker_embedding_meta, 'w', encoding='utf-8') as f:
				for speaker in hparams.speakers:
					f.write('{}\n'.format(speaker))

		speaker_embedding_meta = speaker_embedding_meta.replace(log_dir, '..')

	#book keeping
	step = 0
	time_window = ValueWindow(100)
	loss_window = ValueWindow(100)
	sh_saver = create_shadow_saver(model, global_step)

	log('Wavenet training set to a maximum of {} steps'.format(args.wavenet_train_steps))

	#Memory allocation on the memory
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	run_init = False

	#Train
	with tf.Session(config=config) as sess:
		try:
			summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
			sess.run(tf.global_variables_initializer())

			#saved model restoring
			if args.restore:
				# Restore saved model if the user requested it, default = True
				try:
					checkpoint_state = tf.train.get_checkpoint_state(save_dir)

					if (checkpoint_state and checkpoint_state.model_checkpoint_path):
						log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
						load_averaged_model(sess, sh_saver, checkpoint_state.model_checkpoint_path)
					else:
						log('No model to load at {}'.format(save_dir), slack=True)
						if hparams.wavenet_weight_normalization:
							run_init = True

				except tf.errors.OutOfRangeError as e:
					log('Cannot restore checkpoint: {}'.format(e), slack=True)
			else:
				log('Starting new training!', slack=True)
				if hparams.wavenet_weight_normalization:
					run_init = True

			if run_init:
				log('\nApplying Weight normalization in fresh training. Applying data dependent initialization forward pass..')
				#Create init_model
				init_model, _ = model_train_mode(args, feeder, hparams, global_step, init=True)

			#initializing feeder
			feeder.start_threads(sess)

			if run_init:
				#Run one forward pass for model parameters initialization (make prediction on init_batch)
				_ = sess.run(init_model.tower_y_hat)
				log('Data dependent initialization done. Starting training!')
			
			#Training loop
			while not coord.should_stop() and step < args.wavenet_train_steps:
				start_time = time.time()
				step, loss, opt = sess.run([global_step, model.loss, model.optimize])
				time_window.append(time.time() - start_time)
				loss_window.append(loss)

				message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
					step, time_window.average, loss, loss_window.average)
				log(message, end='\r', slack=(step % args.checkpoint_interval == 0))

				if np.isnan(loss) or loss > 100:
					log('Loss exploded to {:.5f} at step {}'.format(loss, step))
					raise Exception('Loss exploded')

				if step % args.summary_interval == 0:
					log('\nWriting summary at step {}'.format(step))
					summary_writer.add_summary(sess.run(stats), step)

				if step % args.checkpoint_interval == 0 or step == args.wavenet_train_steps:
					save_log(sess, step, model, plot_dir, wav_dir, hparams=hparams, model_name=args.model)
					save_checkpoint(sess, sh_saver, checkpoint_path, global_step)

				if step % args.eval_interval == 0:
					log('\nEvaluating at step {}'.format(step))
					eval_step(sess, step, eval_model, eval_plot_dir, eval_wav_dir, summary_writer=summary_writer , hparams=model._hparams, model_name=args.model)

				if hparams.gin_channels > 0 and (step % args.embedding_interval == 0 or step == args.wavenet_train_steps or step == 1):
					#Get current checkpoint state
					checkpoint_state = tf.train.get_checkpoint_state(save_dir)

					#Update Projector
					log('\nSaving Model Speaker Embeddings visualization..')
					add_embedding_stats(summary_writer, [model.embedding_table.name], [speaker_embedding_meta], checkpoint_state.model_checkpoint_path)
					log('WaveNet Speaker embeddings have been updated on tensorboard!')

			log('Wavenet training complete after {} global steps'.format(args.wavenet_train_steps), slack=True)
			return save_dir

		except Exception as e:
			log('Exiting due to exception: {}'.format(e), slack=True)
			traceback.print_exc()
			coord.request_stop(e)


def wavenet_train(args, log_dir, hparams, input_path):
	return train(log_dir, args, hparams, input_path)
