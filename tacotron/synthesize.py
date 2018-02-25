import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
import tensorflow as tf 
import time


sentences = [
	# From July 8, 2017 New York Times:
	'Scientists at the CERN laboratory say they have discovered a new particle.',
	'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
	'President Trump met with other leaders at the Group of 20 conference.',
	'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
	# From Google's Tacotron example page:
	'Generative adversarial network or variational auto-encoder.',
	'The buses aren\'t the problem, they actually provide a solution.',
	'Does the quick brown fox jump over the lazy dog?',
	'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
]


def run_eval(args, checkpoint_path):
	print(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path)
	for i, text in enumerate(sentences):
		start = time.time()
		synth.synthesize(text, i, args.output_dir)
		print('synthesized sentence n°{} in {:.3f} sec'.format(i+1, time.time()-start))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', default='logs-Tacotron/pretrained/', help='Path to model checkpoint')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
	args = parser.parse_args()
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	hparams.parse(args.hparams)

	os.makedirs(args.output_dir, exist_ok=True)

	try:
		checkpoint_path = tf.train.get_checkpoint_state(args.checkpoint).model_checkpoint_path
		print('loaded model at {}'.format(checkpoint_path))
	except:
		raise AssertionError('Cannot restore checkpoint: {}, did you train a model?'.format(args.checkpoint))

	run_eval(args, checkpoint_path)


if __name__ == '__main__':
	main()
