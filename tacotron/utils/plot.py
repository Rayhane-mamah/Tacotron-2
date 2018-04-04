import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 


def split_title_line(title_text, max_words=5):
	"""
	A function that splits any string based on specific character
	(returning it with the string), with maximum number of words on it
	"""
	seq = title_text.split()
	return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])

def plot_alignment(alignment, path, info=None, split_title=False):
	fig, ax = plt.subplots()
	im = ax.imshow(
		alignment,
		aspect='auto',
		origin='lower',
		interpolation='none')
	fig.colorbar(im, ax=ax)
	xlabel = 'Decoder timestep'
	if info is not None:
		if split_title:
			title = split_title_line(info)
		else:
			title = info
	plt.xlabel(xlabel)
	plt.title(title)
	plt.ylabel('Encoder timestep')
	plt.tight_layout()
	plt.savefig(path, format='png')


def plot_spectrogram(spectrogram, path, info=None, split_title=False):
	plt.figure()
	plt.imshow(np.rot90(spectrogram))
	plt.colorbar(shrink=0.65, orientation='horizontal')
	plt.ylabel('mels')
	xlabel = 'frames'
	if info is not None:
		if split_title:
			title = split_title_line(info)
		else:
			title = info
	plt.xlabel(xlabel)
	plt.title(title)
	plt.tight_layout()
	plt.savefig(path, format='png')
