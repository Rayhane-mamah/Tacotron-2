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

def spectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
	return _normalize(S)

def melspectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(_linear_to_mel(np.abs(D)))
	return _normalize(S)

def find_endpoint(wav, threshhold_db=-40, min_silence_sec=0.8):
	window_length = int(hparams.sample_rate * min_silence_sec)
	hop_length = int(window_length / 4)
	threshhold = _db_to_amp(threshhold_db)
	for x in range(hop_length, len(wav) - window_length, hop_length):
		if np.max(wav[x: x+window_length]) < threshhold:
			return x + hop_length
	return len(wav)

def _stft(y):
	n_fft, hop_length, win_lenght = _stft_params()
	return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_lenght)

def istft(y):
	_, hop_length, win_lenght = _stft_params()
	return librosa.istft(y=y, hop_length=hop_length, win_lenght=win_lenght)

def _stft_params():
	n_fft = (hparams.num_freq - 1) * 2
	hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
	win_lenght = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
	return n_fft, hop_length, win_lenght

# Conversions

_mel_basis = None

def _linear_to_mel(spectogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
	n_fft = (hparams.num_freq - 1) * 2
	return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)

def _amp_to_db(x):
	return 20 * np.log10(np.maximum(1e-5, x))

def _dp_to_amp(x):
	return np.power(10.0, x * 0.05)

def _normalize(S):
	return np.clip((S - hparams.min_level_db) / (-hparams.min_level_db), 0, 1)

def _denormalize(D):
	return (np.clip(D, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
