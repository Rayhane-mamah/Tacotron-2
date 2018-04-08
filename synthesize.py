import argparse
from tacotron.synthesize import tacotron_synthesize


def main():
	accepted_modes = ['eval', 'synthesis']
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', default='logs-Tacotron/pretrained/', help='Path to model checkpoint')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--mode', default='synthesis', help='mode of run: can be one of {}'.format(accepted_modes))
	parser.add_argument('--GTA', default=True, help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
	args = parser.parse_args()
	
	accepted_models = ['Tacotron', 'Wavenet']

	if args.model not in accepted_models:
		raise ValueError('please enter a valid model to train: {}'.format(accepted_models))

	if args.mode not in accepted_modes:
		raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))

	if args.model == 'Tacotron':
		tacotron_synthesize(args)
	elif args.model == 'Wavenet':
		raise NotImplementedError('Wavenet is still a work in progress, thank you for your patience!')


if __name__ == '__main__':
	main()