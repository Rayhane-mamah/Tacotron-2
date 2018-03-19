import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import preprocessor
from hparams import hparams


def preprocess(args):
	in_dir = os.path.join(args.base_dir, args.input)
	out_dir = os.path.join(args.base_dir, args.output)
	os.makedirs(out_dir, exist_ok=True)
	metadata = preprocessor.build_from_path(in_dir, out_dir, args.n_jobs, tqdm=tqdm)
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


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default=os.path.dirname(os.path.realpath(__file__)))
	parser.add_argument('--input', default='LJSpeech-1.1')
	parser.add_argument('--output', default='training')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	preprocess(args)


if __name__ == '__main__':
	main()
