from .wavenet import WaveNet 


def create_model(name, hparams):
	if name == 'WaveNet'
		return WaveNet(hparams)
	else:
		raise Exception('Unknow model: {}'.format(name))