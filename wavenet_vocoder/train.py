import argparse
import sys
import os
from datetime import datetime

from wavenet_vocoder.models import create_model
import numpy as np 

from hparams import hparams, hparams_debug_string



def sanity_check(model, c, g):
	if model.has_speaker_embedding():
		if g is None:
			raise RuntimeError('Wavenet expects speaker embedding, but speaker-id is not defined')
	else:
		if g is not None:
			raise RuntimeError('Wavenet expects no speaker embedding, but speaker-id is provided')

	if model.local_conditioning_enabled():
		if c is None:
			raise RuntimeError('Wavenet expected conditional features, but none were given')
	else:
		if c is not None:
			raise RuntimeError('Wavenet expects no conditional features, but features were given')


