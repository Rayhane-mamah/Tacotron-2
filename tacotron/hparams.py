import tensorflow as tf 
import numpy as np 


# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
	# text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
	cleaners='english_cleaners',

	#Audio
	num_mels=80, 
	num_freq=1025,
	sample_rate=22050, #22050 Hz (corresponding to ljspeech dataset)
	frame_length_ms= 50,
	frame_shift_ms= 12.5,
	lfilter=True, #whether to use preemphasis
	preemphasis=0.97,
	min_level_db=-100,
	ref_level_db=20,
	fmin=125,
	fmax=7600,
	power=1.3,
	griffin_lim_iters=60,

	#Model
	outputs_per_step = 5, #number of frames to generate at each decoding step (speeds up computation and allows for higher batch size)
	embedding_dim = 512, #dimension of embedding space
	enc_conv_num_layers=3, #number of encoder convolutional layers
	enc_conv_kernel_size=(5, ), #size of encoder convolution filters for each layer
	enc_conv_channels=512, #number of encoder convolutions filters for each layer
	encoder_lstm_units=256, #number of lstm units for each direction (forward and backward)
	attention_dim = 128, #dimension of attention space
	prenet_layers=[128, 128], #number of layers and number of units of prenet
	decoder_layers=2, #number of decoder lstm layers
	decoder_lstm_units=512, #number of decoder lstm units on each layer
	postnet_num_layers=5, #number of postnet convolutional layers
	postnet_kernel_size=(5, ), #size of postnet convolution filters for each layer
	postnet_channels=512, #number of postnet convolution filters for each layer
	max_iters=175, #Max decoder steps during inference (feel free to change it)

	#Training
	batch_size = 32, #number of training samples on each training steps
	reg_weight = 10**(-6), #regularization weight (for l2 regularization)
	decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
	decay_steps = 50000, #starting point for learning rate decay (and determines the decay slope)
	decay_rate = 0.4, #learning rate decay rate
	initial_learning_rate = 10**(-3), #starting learning rate
	final_learning_rate = 10**(-5), #minimal learning rate
	adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
	adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
	adam_epsilon = 10**(-6), #AdamOptimizer beta3 parameter
	zoneout_rate=0.1, #zoneout rate for all LSTM cells in the network
	dropout_rate=0.5, #dropout rate for all convolutional layers + prenet

	#Eval sentences
	sentences = [
	# From July 8, 2017 New York Times:
	'Scientists at the CERN laboratory say they have discovered a new particle.',
	'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
	'President Trump met with other leaders at the Group of 20 conference.',
	'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
	# From Google's Tacotron example page:
	'Generative adversarial network or variational auto-encoder.',
	'Basilar membrane and otolaryngology are not auto-correlations.',
	'He has read the whole thing.',
	'He reads books.',
	"Don't desert me here in the desert!",
	'He thought it was time to present the present.',
	'Thisss isrealy awhsome.',
	'Punctuation sensitivity, is working.',
	'Punctuation sensitivity is working.',
	"The buses aren't the problem, they actually provide a solution.",
	"The buses aren't the PROBLEM, they actually provide a SOLUTION.",
	"The quick brown fox jumps over the lazy dog.",
	"Does the quick brown fox jump over the lazy dog?",
	"Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
	"She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
	"The blue lagoon is a nineteen eighty American romance adventure film.",
	"Tajima Airport serves Toyooka.",
	'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
	#From Training data:
	'the rest being provided with barrack beds, and in dimensions varying from thirty feet by fifteen to fifteen feet by ten.',
	'in giltspur street compter, where he was first lodged',
	]

	)

def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
  return 'Hyperparameters:\n' + '\n'.join(hp)