import numpy as np 
from datetime import datetime
import os
import subprocess
import time
import tensorflow as tf 
import traceback
import argparse

from datasets.feeder import Feeder
from hparams import hparams, hparams_debug_string
from models import create_model
from utils.text import sequence_to_text
from utils import audio, infolog, plot, ValueWindow
log = infolog.log


def add_stats(model):
	with tf.variable_scope('stats') as scope:
		tf.summary.histogram('mel_outputs', model.mel_outputs)
		tf.summary.histogram('mel_targets', model.mel_targets)
		tf.summary.scalar('before_loss', model.before_loss)
		tf.summary.scalar('after_loss', model.after_loss)
		tf.summary.scalar('regularization_loss', model.regularization_loss)
		tf.summary.scalar('stop_token_loss', model.stop_token_loss)
		tf.summary.scalar('loss', model.loss)
		tf.summary.scalar('learning_rate', model.learning_rate) #control learning rate decay speed
		gradient_norms = [tf.norm(grad) for grad in model.gradients]
		tf.summary.histogram('gradient_norm', gradient_norms)
		tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)) #visualize gradients (in case of explosion)
		return tf.summary.merge_all()

def time_string():
	return datetime.now().strftime('%Y-%m-%d %H:%M')

def train(log_dir, args):
	save_dir = os.path.join(log_dir, 'pretrained/')
	checkpoint_path = os.path.join(save_dir, 'model.ckpt')
	input_path = os.path.join(args.base_dir, args.input)
	plot_dir = os.path.join(log_dir, 'plots')
	wav_dir = os.path.join(log_dir, 'wavs')
	mel_dir = os.path.join(log_dir, 'mel-spectrograms')
	os.makedirs(plot_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(mel_dir, exist_ok=True)
	log('Checkpoint path: {}'.format(checkpoint_path))
	log('Loading training data from: {}'.format(input_path))
	log('Using model: {}'.format(args.model))
	log(hparams_debug_string())

	#Set up data feeder
	coord = tf.train.Coordinator()
	with tf.variable_scope('datafeeder') as scope:
		feeder = Feeder(coord, input_path, hparams)

	#Set up model:
	step_count = 0
	try:
		#simple text file to keep count of global step
		with open(os.path.join(log_dir, 'step_counter.txt'), 'r') as file:
			step_count = int(file.read())
	except:
		print('no step_counter file found, assuming there is no saved checkpoint')

	global_step = tf.Variable(step_count, name='global_step', trainable=False)
	with tf.variable_scope('model') as scope:
		model = create_model(args.model, hparams)
		model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets)
		model.add_loss()
		model.add_optimizer(global_step)
		stats = add_stats(model)

	#Book keeping
	step = 0
	save_step = 0
	time_window = ValueWindow(100)
	loss_window = ValueWindow(100)
	saver = tf.train.Saver(max_to_keep=5)

	#Memory allocation on the GPU as needed
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	#Train
	with tf.Session(config=config) as sess:
		try:
			summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
			sess.run(tf.global_variables_initializer())

			#saved model restoring
			if args.restore:
				#Restore saved model if the user requested it, Default = True.
				try:
					checkpoint_state = tf.train.get_checkpoint_state(save_dir)
				except tf.errors.OutOfRangeError as e:
					log('Cannot restore checkpoint: {}'.format(e))

			if (checkpoint_state and checkpoint_state.model_checkpoint_path):
				log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
				saver.restore(sess, checkpoint_state.model_checkpoint_path)

			else:
				if not args.restore:
					log('Starting new training!')
				else:
					log('No model to load at {}'.format(save_dir))

			#initiating feeder
			feeder.start_in_session(sess)

			#Training loop
			while not coord.should_stop():
				start_time = time.time()
				step, loss, opt = sess.run([global_step, model.loss, model.optimize])
				time_window.append(time.time() - start_time)
				loss_window.append(loss)
				message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
					step, time_window.average, loss, loss_window.average)
				log(message, end='\r')

				if loss > 100 or np.isnan(loss):
					log('Loss exploded to {:.5f} at step {}'.format(loss, step))
					raise Exception('Loss exploded')

				if step % args.summary_interval == 0:
					log('\nWriting summary at step: {}'.format(step))
					summary_writer.add_summary(sess.run(stats), step)
				
				if step % args.checkpoint_interval == 0:
					with open(os.path.join(log_dir,'step_counter.txt'), 'w') as file:
						file.write(str(step))
					log('Saving checkpoint to: {}-{}'.format(checkpoint_path, step))
					saver.save(sess, checkpoint_path, global_step=step)
					save_step = step
					
					log('Saving alignment, Mel-Spectrograms and griffin-lim inverted waveform..')
					input_seq, prediction, alignment, target = sess.run([model.inputs[0],
							 model.mel_outputs[0],
							 model.alignments[0],
							 model.mel_targets[0],
							 ])
					#save predicted spectrogram to disk (for plot and manual evaluation purposes)
					mel_filename = 'ljspeech-mel-prediction-step-{}.npy'.format(step)
					np.save(os.path.join(mel_dir, mel_filename), prediction.T, allow_pickle=False)

					#save griffin lim inverted wav for debug.
					wav = audio.inv_mel_spectrogram(prediction.T)
					audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-waveform.wav'.format(step)))

					#save alignment plot to disk (control purposes)
					plot.plot_alignment(alignment, os.path.join(plot_dir, 'step-{}-align.png'.format(step)),
						info='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss))
					#save real mel-spectrogram plot to disk (control purposes)
					plot.plot_spectrogram(target, os.path.join(plot_dir, 'step-{}-real-mel-spectrogram.png'.format(step)),
						info='{}, {}, step={}, Real'.format(args.model, time_string(), step, loss))
					#save predicted mel-spectrogram plot to disk (control purposes)
					plot.plot_spectrogram(prediction, os.path.join(plot_dir, 'step-{}-pred-mel-spectrogram.png'.format(step)),
						info='{}, {}, step={}, loss={:.5}'.format(args.model, time_string(), step, loss))
					log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))

		except Exception as e:
			log('Exiting due to exception: {}'.format(e), slack=True)
			traceback.print_exc()
			coord.request_stop(e)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--input', default='training/train.txt')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--name', help='Name of the run, Used for logging, Defaults to model name')
	parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
	parser.add_argument('--summary_interval', type=int, default=10,
		help='Steps between running summary ops')
	parser.add_argument('--checkpoint_interval', type=int, default=100,
		help='Steps between writing checkpoints')
	parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
	args = parser.parse_args()

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
	run_name = args.name or args.model
	log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
	os.makedirs(log_dir, exist_ok=True)
	infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name)
	args.hparams = hparams
	train(log_dir, args)

if __name__ == '__main__':
	main()
