import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from tacotron.datasets import preprocessor
from tacotron.hparams import hparams


def preprocess(args, input_folders, out_dir):
	output_folder = os.path.join(out_dir, 'mels')
	os.makedirs(output_folder, exist_ok=True)
	metadata = preprocessor.build_from_path(input_folders, output_folder, args.n_jobs, tqdm=tqdm)
	write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	frames = sum([int(m[1]) for m in metadata])
	frame_shift_ms = hparams.hop_size / hparams.sample_rate
	hours = frames * frame_shift_ms / 3600
	print('Write {} utterances, {} frames, ({:.2f} hours)'.format(len(metadata), frames, hours))
	print('Max input length: {}'.format(max(len(m[2]) for m in metadata)))
	print('Max output length: {}'.format(max(int(m[1]) for m in metadata)))

def norm_data(args):
	print('Selecting data folders..')
	supported_datasets = ['LJSpeech-1.1', 'M-AILABS']
	if args.dataset not in supported_datasets:
		raise ValueError('dataset value entered {} does not belong to supported datasets: {}'.format(
			args.dataset, supported_datasets))

	if args.dataset == 'LJSpeech-1.1':
		return [os.path.join(args.base_dir, args.dataset)]

	
	if args.dataset == 'M-AILABS':
		supported_languages = ['en_US', 'en_GB', 'fr_FR', 'it_IT', 'de_DE', 'es-ES', 'ru-RU', 
			'uk-UA', 'pl-PL', 'nl-NL', 'pt-PT', 'sv-FI', 'sv-SE', 'tr-TR', 'ar-SA']
		if args.language not in supported_languages:
			raise ValueError('Please enter a supported language to use from M-AILABS dataset! \n{}'.format(
				supported_languages))

		supported_voices = ['female', 'male', 'mix']
		if args.voice not in supported_voices:
			raise ValueError('Please enter a supported voice option to use from M-AILABS dataset! \n{}'.format(
				supported_voices))

		path = os.path.join(args.base_dir, args.language, 'by_book', args.voice)
		supported_readers = [e for e in os.listdir(path) if e != '.DS_Store']
		if args.reader not in supported_readers:
			raise ValueError('Please enter a valid reader for your language and voice settings! \n{}'.format(
				supported_readers))

		path = os.path.join(path, args.reader)
		supported_books = [e for e in os.listdir(path) if e != '.DS_Store']

		if args.merge_books:
			return [os.path.join(path, book) for book in supported_books]

		else:
			if args.book not in supported_books:
				raise ValueError('Please enter a valid book for your reader settings! \n{}'.format(
					supported_books))

			return [os.path.join(path, args.book)]


def tacotron_preprocess(args):
	input_folders = norm_data(args)
	output_folder = os.path.join(args.base_dir, args.output)

	preprocess(args, input_folders, output_folder)
