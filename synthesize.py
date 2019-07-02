import argparse
import os
from warnings import warn

import tensorflow as tf

from hparams import hparams
from infolog import log
from tacotron.synthesize import tacotron_synthesize


def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	run_name = args.name or args.tacotron_name or args.model
	taco_checkpoint = os.path.join('logs-' + run_name, 'taco_' + args.checkpoint)

	run_name = args.name or args.wavenet_name or args.model
	wave_checkpoint = os.path.join('logs-' + run_name, 'wave_' + args.checkpoint)
	return taco_checkpoint, wave_checkpoint, modified_hp

def get_sentences(args):
	if args.text_list != '':
		with open(args.text_list) as f:
			#sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
			i = 0
			sentences = []
			for line in f.readlines():
				i += 1
				if i % 2 == 0:
					sentences.append(line.strip())
	else:
		sentences = hparams.sentences
	return sentences


def main():
	accepted_modes = ['eval', 'synthesis', 'live']
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
	parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
	parser.add_argument('--wavenet_name', help='Name of logging directory of WaveNet. If trained separately')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
	parser.add_argument('--mels_dir', default='tacotron_output/eval/', help='folder to contain mels to synthesize audio from using the Wavenet')
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--mode', default='eval', help='mode of run: can be one of {}'.format(accepted_modes))
	parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
	parser.add_argument('--text_list', default='', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
	parser.add_argument('--speaker_id', default=None, help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')
	args = parser.parse_args()

	if args.mode not in accepted_modes:
		raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))

	if args.GTA not in ('True', 'False'):
		raise ValueError('GTA option must be either True or False')

	taco_checkpoint, wave_checkpoint, hparams = prepare_run(args)
	sentences = get_sentences(args)

	tacotron_synthesize(args, hparams, taco_checkpoint, sentences)


if __name__ == '__main__':
	main()
