import argparse
import sys
import os
from datetime import datetime
import time
import librosa

from wavenet_vocoder.models import create_model
from wavenet_vocoder.feeder import Feeder
from tacotron.utils import ValueWindow
import numpy as np 
from scipy.io import wavfile
import tensorflow as tf
from . import util

from hparams import hparams_debug_string
import infolog

log = infolog.log


def add_train_stats(model):
	with tf.variable_scope('stats') as scope:
		tf.summary.histogram('wav_outputs', model.y_hat)
		tf.summary.histogram('wav_targets', model.y)
		tf.summary.scalar('loss', model.loss)
		return tf.summary.merge_all()

def add_test_stats(summary_writer, step, eval_loss):
	values = [
	tf.Summary.Value(tag='eval_model/eval_stats/eval_loss'),
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
	return tf.train.Saver(shadow_dict, max_to_keep=5)

def load_averaged_model(sess, sh_saver, checkpoint_path):
	sh_saver.restore(sess, checkpoint_path)


def eval_step(sess, global_step, model, plot_dir, audio_dir, summary_writer, hparams):
	'''Evaluate model during training.
	Supposes that model variables are averaged.
	'''
	start_time = time.time()
	y_hat, y_target, loss = sess.run([model.y_hat, model.y_target, model.eval_loss])
	duration = time.time() - start_time
	log('Time Evaluation: Generation of {} audio frames took {:.3f} sec ({:.3f} frames/sec)'.format(
		len(y_target), duration, len(y_target)/duration))

	pred_wav_path = os.path.join(audio_dir, 'step-{}-pred.wav'.format(global_step))
	target_wav_path = os.path.join(audio_dir, 'step-{}-real.wav'.format(global_step))
	plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(global_step))

	#Save Audio
	wavfile.write(pred_wav_path, hparams.sample_rate, y_hat)
	wavfile.write(target_wav_path, hparams.sample_rate, y_target)

	#Save figure
	util.waveplot(plot_path, y_hat, y_target, model._hparams)
	log('Eval loss for global step {}: {:.3f}'.format(global_step, loss))

	log('Writing eval summary!')
	add_test_stats(summary_writer, global_step, loss)

def save_log(sess, global_step, model, plot_dir, audio_dir, hparams):
	log('\nSaving intermediate states at step {}'.format(global_step))
	idx = 0
	y_hat, y, length = sess.run([model.y_hat_log[idx], model.y_log[idx], model.input_lengths[idx]])

	#mask by length
	y_hat[length:] = 0
	y[length:] = 0

	#Make audio and plot paths
	pred_wav_path = os.path.join(audio_dir, 'step-{}-pred.wav'.format(global_step))
	target_wav_path = os.path.join(audio_dir, 'step-{}-real.wav'.format(global_step))
	plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(global_step))

	#Save audio
	librosa.output.write_wav(pred_wav_path, y_hat, sr=hparams.sample_rate)
	librosa.output.write_wav(target_wav_path, y, sr=hparams.sample_rate)

	#Save figure
	util.waveplot(plot_path, y_hat, y, hparams)

def save_checkpoint(sess, saver, checkpoint_path, global_step):
	saver.save(sess, checkpoint_path, global_step=global_step)


def model_train_mode(args, feeder, hparams, global_step):
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
		model_name = None
		if args.model in ('Tacotron-2', 'Both'):
			model_name = 'WaveNet'
		model = create_model(model_name or args.model, hparams)
		#initialize model to train mode
		model.initialize(feeder.targets, feeder.local_condition_features, feeder.global_condition_features,
			feeder.input_lengths, x=feeder.inputs)
		model.add_loss()
		model.add_optimizer(global_step)
		stats = add_train_stats(model)
		return model, stats

def model_test_mode(args, feeder, hparams, global_step):
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
		model_name = None
		if args.model in ('Tacotron-2', 'Both'):
			model_name = 'WaveNet'
		model = create_model(model_name or args.model, hparams)
		#initialize model to test mode
		model.initialize(feeder.eval_targets, feeder.eval_local_condition_features, feeder.eval_global_condition_features,
			feeder.eval_input_lengths)
		model.add_loss()
		return model

def train(log_dir, args, hparams, input_path):
	save_dir = os.path.join(log_dir, 'pretrained/')
	eval_dir = os.path.join(log_dir, 'eval-dir')
	audio_dir = os.path.join(log_dir, 'wavs')
	plot_dir = os.path.join(log_dir, 'plots')
	wav_dir = os.path.join(log_dir, 'wavs')
	eval_audio_dir = os.path.join(eval_dir, 'wavs')
	eval_plot_dir = os.path.join(eval_dir, 'plots')
	checkpoint_path = os.path.join(save_dir, 'model.ckpt')
	input_path = os.path.join(args.base_dir, input_path)
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(audio_dir, exist_ok=True)
	os.makedirs(plot_dir, exist_ok=True)
	os.makedirs(eval_audio_dir, exist_ok=True)
	os.makedirs(eval_plot_dir, exist_ok=True)

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

	#book keeping
	step = 0
	time_window = ValueWindow(100)
	loss_window = ValueWindow(100)
	sh_saver = create_shadow_saver(model, global_step)

	log('Wavenet training set to a maximum of {} steps'.format(args.wavenet_train_steps))

	#Memory allocation on the memory
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	#Train
	with tf.Session(config=config) as sess:
		try:
			summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
			sess.run(tf.global_variables_initializer())

			#saved model restoring
			if args.restore:
				#Restore saved model if the user requested it, default = True
				try:
					checkpoint_state = tf.train.get_checkpoint_state(save_dir)
				except tf.errors.OutOfRangeError as e:
					log('Cannot restore checkpoint: {}'.format(e))

			if (checkpoint_state and checkpoint_state.model_checkpoint_path):
				log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
				load_averaged_model(sess, sh_saver, checkpoint_state.model_checkpoint_path)

			else:
				if not args.restore:
					log('Starting new training!')
				else:
					log('No model to load at {}'.format(save_dir))

			#initializing feeder
			feeder.start_threads(sess)

			#Training loop
			while not coord.should_stop() and step < args.wavenet_train_steps:
				start_time = time.time()
				step, y_hat, loss, opt = sess.run([global_step, model.y_hat, model.loss, model.optimize])
				time_window.append(time.time() - start_time)
				loss_window.append(loss)

				message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
					step, time_window.average, loss, loss_window.average)
				log(message, end='\r')

				if loss > 100 or np.isnan(loss):
					log('Loss exploded to {:.5f} at step {}'.format(loss, step))
					raise Exception('Loss exploded')

				if step % args.summary_interval == 0:
					log('\nWriting summary at step {}'.format(step))
					summary_writer.add_summary(sess.run(stats), step)

				if step % args.checkpoint_interval == 0:
					save_log(sess, step, model, plot_dir, audio_dir, hparams=hparams)
					save_checkpoint(sess, sh_saver, checkpoint_path, global_step)

				if step % args.eval_interval == 0:
					log('\nEvaluating at step {}'.format(step))
					eval_step(sess, step, eval_model, eval_plot_dir, eval_audio_dir, summary_writer=summary_writer , hparams=model._hparams)

			log('Wavenet training complete after {} global steps'.format(args.wavenet_train_steps))

		except Exception as e:
			log('Exiting due to Exception: {}'.format(e))


def wavenet_train(args, log_dir, hparams, input_path):
	train(log_dir, args, hparams, input_path)
