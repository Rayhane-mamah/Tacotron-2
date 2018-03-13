import librosa
import librosa.filters
import numpy as np 
from scipy import signal
from hparams import hparams
import tensorflow as tf 


def load_wav(path):
	return librosa.core.load(path, sr=hparams.sample_rate)[0]

def save_wav(wav, path):
	wav *= 32767 / max(0.01, np.max(np.abs(wav))) 
	librosa.output.write_wav(path, wav.astype(np.int16), hparams.sample_rate)

def preemphasis(x):
	return signal.lfilter([1, -hparams.preemphasis], [1], x)

def inv_preemphasis(x):
	return signal.lfilter([1], [1, -hparams.preemphasis], x)

def melspectrogram(y):
	if hparams.lfilter:
		D = _stft(preemphasis(y))
	else:
		D = _stft(y)
	S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
	return _normalize(S)

def inv_mel_spectrogram(mel_spectrogram):
	'''Converts mel spectrogram to waveform using librosa'''
	S = _mel_to_linear(_db_to_amp(_denormalize(mel_spectrogram) + hparams.ref_level_db))  # Convert back to linear
	if hparams.lfilter:
		return inv_preemphasis(_griffin_lim(S ** hparams.power)) # Reconstruct phase
	return _griffin_lim(S ** hparams.power)

def _griffin_lim(S):
	'''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles)
	for i in range(hparams.griffin_lim_iters):
		angles = np.exp(1j * np.angle(_stft(y)))
		y = _istft(S_complex * angles)
	return y

def _stft(y):
	n_fft, hop_length, win_length = _stft_params()
	return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _istft(y):
	_, hop_length, win_length = _stft_params()
	return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def _stft_params():
	n_fft = (hparams.num_freq - 1) * 2
	hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
	win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
	return n_fft, hop_length, win_length


# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram):
	global _inv_mel_basis
	if _inv_mel_basis is None:
		_inv_mel_basis = np.linalg.pinv(_build_mel_basis())
	return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis():
	n_fft = (hparams.num_freq - 1) * 2
	return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels,
							   fmin=hparams.fmin, fmax=hparams.fmax,)

def _amp_to_db(x):
	return 20 * np.log10(np.maximum(1e-5, x)) + 0.01

def _db_to_amp(x):
	return np.power(10.0, (x - 0.01) * 0.05)

def _normalize(S):
	return np.clip((S - hparams.min_level_db) / (-hparams.min_level_db), 0, 1)

def _denormalize(D):
	return (np.clip(D, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
