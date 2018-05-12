import librosa
import librosa.filters
import numpy as np
from scipy import signal
from tacotron.hparams import hparams
import tensorflow as tf


def load_wav(path):
	return librosa.core.load(path, sr=hparams.sample_rate)[0]

def save_wav(wav, path):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	librosa.output.write_wav(path, wav.astype(np.int16), hparams.sample_rate)

def trim_silence(wav):
	'''Trim leading and trailing silence

	Useful for M-AILABS dataset if we choose to trim the extra 0.5 silences.
	'''
	return librosa.effects.trim(wav)[0]

def preemphasis(x):
	return signal.lfilter([1, -hparams.preemphasis], [1], x)

def inv_preemphasis(x):
	return signal.lfilter([1], [1, -hparams.preemphasis], x)

def get_hop_size():
	hop_size = hparams.hop_size
	if hop_size is None:
		assert hparams.frame_shift_ms is not None
		hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
	return hop_size

def melspectrogram(wav):
	D = _stft(wav)
	S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db

	if hparams.mel_normalization:
		return _normalize(S)
	return S


def inv_mel_spectrogram(mel_spectrogram):
	'''Converts mel spectrogram to waveform using librosa'''
	if hparams.mel_normalization:
		D = _denormalize(mel_spectrogram)
	else:
		D = mel_spectrogram

	S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db))  # Convert back to linear

	if hparams.use_lws:
		processor = _lws_processor()
		D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
		y = processor.istft(D).astype(np.float32)
		return y
	else:
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

def _lws_processor():
	import lws
	return lws.lws(hparams.fft_size, get_hop_size(), mode="speech")

def _stft(y):
	if hparams.use_lws:
		return _lws_processor().stft(y).T
	else:
		return librosa.stft(y=y, n_fft=hparams.fft_size, hop_length=get_hop_size())

def _istft(y):
	return librosa.istft(y, hop_length=get_hop_size())


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
	assert hparams.fmax <= hparams.sample_rate // 2
	return librosa.filters.mel(hparams.sample_rate, hparams.fft_size, n_mels=hparams.num_mels,
							   fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x):
	min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
	return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
	return np.power(10.0, (x) * 0.05)

def _normalize(S):
	if hparams.allow_clipping_in_normalization:
		if hparams.symmetric_mels:
			return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
			 -hparams.max_abs_value, hparams.max_abs_value)
		else:
			return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)

	assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
	if hparams.symmetric_mels:
		return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
	else:
		return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

def _denormalize(D):
	if hparams.allow_clipping_in_normalization:
		if hparams.symmetric_mels:
			return (((np.clip(D, -hparams.max_abs_value,
				hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
				+ hparams.min_level_db)
		else:
			return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

	if hparams.symmetric_mels:
		return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
	else:
		return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
