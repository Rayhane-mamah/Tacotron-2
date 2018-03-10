import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 


def plot_alignment(alignment, path, info=None):
	fig, ax = plt.subplots()
	im = ax.imshow(
		alignment,
		aspect='auto',
		origin='lower',
		interpolation='none')
	fig.colorbar(im, ax=ax)
	xlabel = 'Decoder timestep'
	if info is not None:
		xlabel += '\n\n' + info
	plt.xlabel(xlabel)
	plt.ylabel('Encoder timestep')
	plt.tight_layout()
	plt.savefig(path, format='png')


def plot_spectrogram(spectrogram, path, info=None):
	plt.figure()
	plt.imshow(np.rot90(spectrogram))
	plt.colorbar(shrink=0.5, orientation='horizontal')
	plt.ylabel('mels')
	xlabel = 'frames'
	if info is not None:
		xlabel += '\n' + info
	plt.xlabel(xlabel)
	plt.tight_layout()
	plt.savefig(path, format='png')
