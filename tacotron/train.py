import argparse
import os
import subprocess
import time
import traceback
from datetime import datetime
import infolog
import numpy as np
import tensorflow as tf
from datasets import audio
from hparams import hparams_debug_string
from tacotron.feeder import Feeder
from tacotron.models import create_model
from tacotron.utils import ValueWindow, plot
from tacotron.utils.text import sequence_to_text
from tqdm import tqdm

log = infolog.log


def add_train_stats(model, hparams):
	with tf.variable_scope('stats') as scope:
		tf.summary.histogram('mel_outputs', model.mel_outputs)
		tf.summary.histogram('mel_targets', model.mel_targets)
		tf.summary.scalar('before_loss', model.before_loss)
		tf.summary.scalar('after_loss', model.after_loss)
		if hparams.predict_linear:
			tf.summary.scalar('linear_loss', model.linear_loss)
		tf.summary.scalar('regularization_loss', model.regularization_loss)
		tf.summary.scalar('stop_token_loss', model.stop_token_loss)
		tf.summary.scalar('attention_loss', model.attention_loss)
		tf.summary.scalar('loss', model.loss)
		tf.summary.scalar('learning_rate', model.learning_rate) #Control learning rate decay speed
		if hparams.tacotron_teacher_forcing_mode == 'scheduled':
			tf.summary.scalar('teacher_forcing_ratio', model.ratio) #Control teacher forcing ratio decay when mode = 'scheduled'
		gradient_norms = [tf.norm(grad) for grad in model.gradients]
		tf.summary.histogram('gradient_norm', gradient_norms)
		tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)) #visualize gradients (in case of explosion)
		return tf.summary.merge_all()

def add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, stop_token_loss, attention_loss, loss):
	values = [
	tf.Summary.Value(tag='eval_model/eval_stats/eval_before_loss', simple_value=before_loss),
	tf.Summary.Value(tag='eval_model/eval_stats/eval_after_loss', simple_value=after_loss),
	tf.Summary.Value(tag='eval_model/eval_stats/stop_token_loss', simple_value=stop_token_loss),
	tf.Summary.Value(tag='eval_model/eval_stats/attention_loss', simple_value=attention_loss),
	tf.Summary.Value(tag='eval_model/eval_stats/eval_loss', simple_value=loss),
	]
	if linear_loss is not None:
		values.append(tf.Summary.Value(tag='eval_model/eval_stats/eval_linear_loss', simple_value=linear_loss))
	test_summary = tf.Summary(value=values)
	summary_writer.add_summary(test_summary, step)

def time_string():
	return datetime.now().strftime('%Y-%m-%d %H:%M')

def model_train_mode(args, feeder, hparams, global_step):
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
		model_name = None
		if args.model == 'Tacotron-2':
			model_name = 'Tacotron'
		model = create_model(model_name or args.model, hparams)
		if hparams.predict_linear:
			model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets, linear_targets=feeder.linear_targets,
				targets_lengths=feeder.targets_lengths, global_step=global_step,
				is_training=True)
		else:
			model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets,
				targets_lengths=feeder.targets_lengths, global_step=global_step,
				is_training=True)
		model.add_loss()
		model.add_optimizer(global_step)
		stats = add_train_stats(model, hparams)
		return model, stats

def model_test_mode(args, feeder, hparams, global_step):
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
		model_name = None
		if args.model == 'Tacotron-2':
			model_name = 'Tacotron'
		model = create_model(model_name or args.model, hparams)
		if hparams.predict_linear:
			model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets, feeder.eval_token_targets,
				linear_targets=feeder.eval_linear_targets, targets_lengths=feeder.eval_targets_lengths, global_step=global_step,
				is_training=False, is_evaluating=True)
		else:
			model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets, feeder.eval_token_targets,
				targets_lengths=feeder.eval_targets_lengths, global_step=global_step, is_training=False, is_evaluating=True)
		model.add_loss()
		return model

def train(log_dir, args, hparams):
	save_dir = os.path.join(log_dir, 'taco_pretrained')
	plot_dir = os.path.join(log_dir, 'plots')
	wav_dir = os.path.join(log_dir, 'wavs')
	mel_dir = os.path.join(log_dir, 'mel-spectrograms')
	eval_dir = os.path.join(log_dir, 'eval-dir')
	eval_plot_dir = os.path.join(eval_dir, 'plots')
	eval_wav_dir = os.path.join(eval_dir, 'wavs')
	tensorboard_dir = os.path.join(log_dir, 'tacotron_events')
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(plot_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(eval_plot_dir, exist_ok=True)
	os.makedirs(eval_wav_dir, exist_ok=True)
	os.makedirs(tensorboard_dir, exist_ok=True)

	checkpoint_path = os.path.join(save_dir, 'tacotron_model.ckpt')
	input_path = os.path.join(args.base_dir, args.tacotron_input)

	if hparams.predict_linear:
		linear_dir = os.path.join(log_dir, 'linear-spectrograms')
		os.makedirs(linear_dir, exist_ok=True)

	log('Checkpoint path: {}'.format(checkpoint_path))
	log('Loading training data from: {}'.format(input_path))
	log('Using model: {}'.format(args.model))
	log(hparams_debug_string())

	#Start by setting a seed for repeatability
	tf.set_random_seed(hparams.tacotron_random_seed)

	#Set up data feeder
	coord = tf.train.Coordinator()
	with tf.variable_scope('datafeeder') as scope:
		feeder = Feeder(coord, input_path, hparams)

	#Set up model:
	global_step = tf.Variable(0, name='global_step', trainable=False)
	model, stats = model_train_mode(args, feeder, hparams, global_step)
	eval_model = model_test_mode(args, feeder, hparams, global_step)

	#Book keeping
	step = 0
	time_window = ValueWindow(100)
	loss_window = ValueWindow(100)
	saver = tf.train.Saver(max_to_keep=1)

	log('Tacotron training set to a maximum of {} steps'.format(args.tacotron_train_steps))

	#Memory allocation on the GPU as needed
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

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
						saver.restore(sess, checkpoint_state.model_checkpoint_path)
					else:
						log('No model to load at {}'.format(save_dir), slack=True)

				except tf.errors.OutOfRangeError as e:
					log('Cannot restore checkpoint: {}'.format(e), slack=True)
			else:
				log('Starting new training!', slack=True)

			#initializing feeder
			feeder.start_threads(sess)

			#Training loop
			while not coord.should_stop() and step < args.tacotron_train_steps:
				start_time = time.time()
				step, loss, opt = sess.run([global_step, model.loss, model.optimize])
				time_window.append(time.time() - start_time)
				loss_window.append(loss)
				message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
					step, time_window.average, loss, loss_window.average)
				log(message, end='\r', slack=(step % args.checkpoint_interval == 0))

				if loss > 100 or np.isnan(loss):
					log('Loss exploded to {:.5f} at step {}'.format(loss, step))
					raise Exception('Loss exploded')

				if step % args.summary_interval == 0:
					log('\nWriting summary at step {}'.format(step))
					summary_writer.add_summary(sess.run(stats), step)

				if step % args.eval_interval == 0:
					#Run eval and save eval stats
					log('\nRunning evaluation at step {}'.format(step))

					eval_losses = []
					before_losses = []
					after_losses = []
					stop_token_losses = []
					attention_losses = []
					linear_losses = []
					linear_loss = None

					if hparams.predict_linear:
						for i in tqdm(range(feeder.test_steps)):
							eloss, before_loss, after_loss, stop_token_loss, linear_loss, attention_loss, mel_p, mel_t, t_len, align, lin_p = sess.run(
								[eval_model.loss, eval_model.before_loss, eval_model.after_loss, eval_model.stop_token_loss,
								eval_model.linear_loss, eval_model.attention_loss, eval_model.mel_outputs[0], eval_model.mel_targets[0],
								eval_model.targets_lengths[0], eval_model.alignments[0], eval_model.linear_outputs[0]])
							eval_losses.append(eloss)
							before_losses.append(before_loss)
							after_losses.append(after_loss)
							stop_token_losses.append(stop_token_loss)
							attention_losses.append(attention_loss)
							linear_losses.append(linear_loss)
						linear_loss = sum(linear_losses) / len(linear_losses)

						wav = audio.inv_linear_spectrogram(lin_p.T, hparams)
						audio.save_wav(wav, os.path.join(eval_wav_dir, 'step-{}-eval-waveform-linear.wav'.format(step)), hparams)
					else:
						for i in tqdm(range(feeder.test_steps)):
							eloss, before_loss, after_loss, stop_token_loss, attention_loss, mel_p, mel_t, t_len, align = sess.run(
								[eval_model.loss, eval_model.before_loss, eval_model.after_loss, eval_model.stop_token_loss,
								eval_model.attention_loss, eval_model.mel_outputs[0], eval_model.mel_targets[0],
								eval_model.targets_lengths[0], eval_model.alignments[0]])
							eval_losses.append(eloss)
							before_losses.append(before_loss)
							after_losses.append(after_loss)
							stop_token_losses.append(stop_token_loss)
							attention_losses.append(attention_loss)

					eval_loss = sum(eval_losses) / len(eval_losses)
					before_loss = sum(before_losses) / len(before_losses)
					after_loss = sum(after_losses) / len(after_losses)
					stop_token_loss = sum(stop_token_losses) / len(stop_token_losses)
					attention_loss = sum(attention_losses) / len(attention_losses)

					log('Saving eval log to {}..'.format(eval_dir))
					# #Save some log to monitor model improvement on same unseen sequence
					wav = audio.inv_mel_spectrogram(mel_p.T, hparams)
					audio.save_wav(wav, os.path.join(eval_wav_dir, 'step-{}-eval-waveform-mel.wav'.format(step)), hparams)

					plot.plot_alignment(align, os.path.join(eval_plot_dir, 'step-{}-eval-align.png'.format(step)),
						info='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, eval_loss),
						max_len=t_len // hparams.outputs_per_step)
					plot.plot_spectrogram(mel_p, os.path.join(eval_plot_dir, 'step-{}-eval-mel-spectrogram.png'.format(step)),
						info='{}, {}, step={}, loss={:.5}'.format(args.model, time_string(), step, eval_loss), target_spectrogram=mel_t,
						max_len=t_len)

					log('Eval loss for global step {}: {:.3f}'.format(step, eval_loss))
					log('Writing eval summary!')
					add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, stop_token_loss, attention_loss, eval_loss)


				if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps:
					#Save model and current global step
					saver.save(sess, checkpoint_path, global_step=global_step)

					log('\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..')
					if hparams.predict_linear:
						input_seq, mel_prediction, linear_prediction, alignment, target, target_length = sess.run([
							model.inputs[0],
							model.mel_outputs[0],
							model.linear_outputs[0],
							model.alignments[0],
							model.mel_targets[0],
							model.targets_lengths[0],
							])

						#save predicted linear spectrogram to disk (debug)
						linear_filename = 'linear-prediction-step-{}.npy'.format(step)
						np.save(os.path.join(linear_dir, linear_filename), linear_prediction.T, allow_pickle=False)

						#save griffin lim inverted wav for debug (linear -> wav)
						wav = audio.inv_linear_spectrogram(linear_prediction.T, hparams)
						audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-linear.wav'.format(step)), hparams)

					else:
						input_seq, mel_prediction, alignment, target, target_length = sess.run([model.inputs[0],
							model.mel_outputs[0],
							model.alignments[0],
							model.mel_targets[0],
							model.targets_lengths[0],
							])

					#save predicted mel spectrogram to disk (debug)
					mel_filename = 'mel-prediction-step-{}.npy'.format(step)
					np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T, allow_pickle=False)

					#save griffin lim inverted wav for debug (mel -> wav)
					wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
					audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-mel.wav'.format(step)), hparams)

					#save alignment plot to disk (control purposes)
					plot.plot_alignment(alignment, os.path.join(plot_dir, 'step-{}-align.png'.format(step)),
						info='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss),
						max_len=target_length // hparams.outputs_per_step)
					#save real and predicted mel-spectrogram plot to disk (control purposes)
					plot.plot_spectrogram(mel_prediction, os.path.join(plot_dir, 'step-{}-mel-spectrogram.png'.format(step)),
						info='{}, {}, step={}, loss={:.5}'.format(args.model, time_string(), step, loss), target_spectrogram=target,
						max_len=target_length)
					log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))

			log('Tacotron training complete after {} global steps!'.format(args.tacotron_train_steps), slack=True)
			return save_dir

		except Exception as e:
			log('Exiting due to exception: {}'.format(e), slack=True)
			traceback.print_exc()
			coord.request_stop(e)

def tacotron_train(args, log_dir, hparams):
	return train(log_dir, args, hparams)
