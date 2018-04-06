import argparse
from tacotron.preprocess import tacotron_preprocess
from multiprocessing import cpu_count


def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--dataset', default='LJSpeech-1.1')
	parser.add_argument('--language', default='en_US')
	parser.add_argument('--voice', default='female')
	parser.add_argument('--reader', default='mary_ann')
	parser.add_argument('--merge_books', type=bool, default=False)
	parser.add_argument('--book', default='northandsouth')
	parser.add_argument('--output', default='training_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	accepted_models = ['Tacotron', 'Wavenet']

	if args.model not in accepted_models:
		raise ValueError('please enter a valid model to train: {}'.format(accepted_models))

	if args.model == 'Tacotron':
		tacotron_preprocess(args)
	elif args.model == 'Wavenet':
		raise NotImplementedError('Wavenet is still a work in progress, thank you for your patience!')


if __name__ == '__main__':
	main()