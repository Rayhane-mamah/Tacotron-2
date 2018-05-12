import argparse
from tacotron.train import tacotron_train


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--input', default='training_data/train.txt')
	parser.add_argument('--name', help='Name of logging directory.')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
	parser.add_argument('--summary_interval', type=int, default=100,
		help='Steps between running summary ops')
	parser.add_argument('--checkpoint_interval', type=int, default=500,
		help='Steps between writing checkpoints')
	parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
	args = parser.parse_args()

	accepted_models = ['Tacotron', 'Wavenet']

	if args.model not in accepted_models:
		raise ValueError('please enter a valid model to train: {}'.format(accepted_models))

	if args.model == 'Tacotron':
		tacotron_train(args)
	elif args.model == 'Wavenet':
		raise NotImplementedError('Wavenet is still a work in progress, thank you for your patience!')


if __name__ == '__main__':
	main()