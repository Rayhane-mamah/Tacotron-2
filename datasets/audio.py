import librosa
import librosa.filters
import numpy as np 
from scipy import signal
import tensorflow as tf 
from scipy.io import wavfile


def load_wav(path, sr):
	return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
	wav *= 32767 / max(0.01, np.max(np.abs(wav))) 
	#proposed by @dsmiller
	wavfile.write(path, sr, wav.astype(np.int16))

#From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
	for start in range(quantized.size):
		if abs(quantized[start] - 127) > silence_threshold:
			break
	for end in range(quantized.size - 1, 1, -1):
		if abs(quantized[end] - 127) > silence_threshold:
			break

	assert abs(quantized[start] - 127) > silence_threshold
	assert abs(quantized[end] - 127) > silence_threshold

	return start, end

def trim_silence(wav, hparams):
	'''Trim leading and trailing silence

	Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
	'''
	#Thanks @begeekmyfriend for pointing out the params contradiction. Should we set separate params for this function?
	return librosa.effects.trim(wav, frame_length=hparams.fft_size, hop_length=get_hop_size(hparams))[0]

def get_hop_size(hparams):
	hop_size = hparams.hop_size
	if hop_size is None:
		assert hparams.frame_shift_ms is not None
		hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
	return hop_size

def linearspectrogram(wav, hparams):
	D = _stft(wav, hparams)
	S = _amp_to_db(np.abs(D), hparams) - hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize(S, hparams)
	return S

def melspectrogram(wav, hparams):
	D = _stft(wav, hparams)
	S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize(S, hparams)
	return S

def inv_linear_spectrogram(linear_spectrogram, hparams):
	'''Converts linear spectrogram to waveform using librosa'''
	if hparams.signal_normalization:
		D = _denormalize(linear_spectrogram, hparams)
	else:
		D = linear_spectrogram

	S = _db_to_amp(D + hparams.ref_level_db) #Convert back to linear

	if hparams.use_lws:
		processor = _lws_processor(hparams)
		D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
		y = processor.istft(D).astype(np.float32)
		return y
	else:
		return _griffin_lim(S ** hparams.power)
	

def inv_mel_spectrogram(mel_spectrogram, hparams):
	'''Converts mel spectrogram to waveform using librosa'''
	if hparams.signal_normalization:
		D = _denormalize(mel_spectrogram, hparams)
	else:
		D = mel_spectrogram

	S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)  # Convert back to linear

	if hparams.use_lws:
		processor = _lws_processor(hparams)
		D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
		y = processor.istft(D).astype(np.float32)
		return y
	else:
		return _griffin_lim(S ** hparams.power)

def _lws_processor(hparams):
	import lws
	return lws.lws(hparams.fft_size, get_hop_size(hparams), mode="speech")

def _griffin_lim(S, hparams):
	'''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles, hparams)
	for i in range(hparams.griffin_lim_iters):
		angles = np.exp(1j * np.angle(_stft(y, hparams)))
		y = _istft(S_complex * angles, hparams)
	return y

def _stft(y, hparams):
	if hparams.use_lws:
		return _lws_processor(hparams).stft(y).T
	else:
		return librosa.stft(y=y, n_fft=hparams.fft_size, hop_length=get_hop_size(hparams))

def _istft(y, hparams):
	return librosa.istft(y, hop_length=get_hop_size(hparams))

def num_frames(length, fsize, fshift):
	"""Compute number of time frames of spectrogram
	"""
	pad = (fsize - fshift)
	if length % fshift == 0:
		M = (length + pad * 2 - fsize) // fshift + 1
	else:
		M = (length + pad * 2 - fsize) // fshift + 2
	return M


def pad_lr(x, fsize, fshift):
	"""Compute left and right padding
	"""
	M = num_frames(len(x), fsize, fshift)
	pad = (fsize - fshift)
	T = len(x) + 2 * pad
	r = (M - 1) * fshift + fsize - T
	return pad, pad + r


# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, hparams):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis(hparams)
	return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, hparams):
	global _inv_mel_basis
	if _inv_mel_basis is None:
		_inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
	return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hparams):
	assert hparams.fmax <= hparams.sample_rate // 2
	return librosa.filters.mel(hparams.sample_rate, hparams.fft_size, n_mels=hparams.num_mels,
							   fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x, hparams):
	min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
	return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
	return np.power(10.0, (x) * 0.05)

def _normalize(S, hparams):
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

def _denormalize(D, hparams):
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