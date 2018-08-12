import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
	# text, you may want to use "basic_cleaners" or "transliteration_cleaners".
	cleaners='english_cleaners',

	#Hardware setup (TODO: multi-GPU parallel tacotron training)
	use_all_gpus = False, #Whether to use all GPU resources. If True, total number of available gpus will override num_gpus.
	num_gpus = 1, #Determines the number of gpus in use
	###########################################################################################################################################

	#Audio
	num_mels = 80, #Number of mel-spectrogram channels and local conditioning dimensionality
	num_freq = 1025, # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
	rescale = True, #Whether to rescale audio prior to preprocessing
	rescaling_max = 0.999, #Rescaling value
	trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	clip_mels_length = True, #For cases of OOM (Not really recommended, working on a workaround)
	max_mel_frames = 1300,  #Only relevant when clip_mels_length = True

	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False,
	silence_threshold=2, #silence threshold used for sound trimming for wavenet preprocessing

	#Mel spectrogram
	n_fft = 2048, #Extra window size is filled with 0 paddings to match this parameter
	hop_size = 300, #For 22050Hz, 275 ~= 12.5 ms
	win_size = 1200, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft)
	sample_rate = 24000, #22050 Hz (corresponding to ljspeech dataset)
	frame_shift_ms = None,

	#M-AILABS (and other datasets) trim params
	trim_fft_size = 512,
	trim_hop_size = 128,
	trim_top_db = 23,

	#Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True,
	allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
	symmetric_mels = False, #Whether to scale the data to be symmetric around 0
	max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max]
	normalize_for_wavenet = True, #whether to rescale to [0, 1] for wavenet.

	#Limits
	min_level_db = -100,
	ref_level_db = 20,
	fmin = 0, #Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
	fmax = 7600,

	#Griffin Lim
	power = 1.5,
	griffin_lim_iters = 60,
	###########################################################################################################################################

	#Tacotron
	outputs_per_step = 2, #number of frames to generate at each decoding step (speeds up computation and allows for higher batch size)
	stop_at_any = True, #Determines whether the decoder should stop when predicting <stop> to any frame or to all of them

	embedding_dim = 512, #dimension of embedding space

	enc_conv_num_layers = 3, #number of encoder convolutional layers
	enc_conv_kernel_size = (5, ), #size of encoder convolution filters for each layer
	enc_conv_channels = 512, #number of encoder convolutions filters for each layer
	encoder_lstm_units = 256, #number of lstm units for each direction (forward and backward)

	smoothing = False, #Whether to smooth the attention normalization function
	attention_dim = 128, #dimension of attention space
	attention_filters = 32, #number of attention convolution filters
	attention_kernel = (31, ), #kernel size of attention convolution
	cumulative_weights = True, #Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

	prenet_layers = [256, 256], #number of layers and number of units of prenet
	decoder_layers = 2, #number of decoder lstm layers
	decoder_lstm_units = 1024, #number of decoder lstm units on each layer
	max_iters = 1000, #Max decoder steps during inference (Just for safety from infinite loop cases)

	postnet_num_layers = 5, #number of postnet convolutional layers
	postnet_kernel_size = (5, ), #size of postnet convolution filters for each layer
	postnet_channels = 512, #number of postnet convolution filters for each layer

	mask_encoder = False, #whether to mask encoder padding while computing attention
	mask_decoder = False, #Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)

	cross_entropy_pos_weight = 1, #Use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1 = disabled)
	predict_linear = True, #Whether to add a post-processing network to the Tacotron to predict linear spectrograms (True mode Not tested!!)
	###########################################################################################################################################


	#Wavenet
	# Input type:
	# 1. raw [-1, 1]
	# 2. mulaw [-1, 1]
	# 3. mulaw-quantize [0, mu]
	# If input_type is raw or mulaw, network assumes scalar input and
	# discretized mixture of logistic distributions output, otherwise one-hot
	# input and softmax output are assumed.
	input_type="raw",
	quantize_channels=2 ** 16,  # 65536 (16-bit) (raw) or 256 (8-bit) (mulaw or mulaw-quantize) // number of classes = 256 <=> mu = 255

	log_scale_min=float(np.log(1e-14)), #Mixture of logistic distributions minimal log scale
	log_scale_min_gauss = float(np.log(1e-7)), #Gaussian distribution minimal allowed log scale

	#To use Gaussian distribution as output distribution instead of mixture of logistics sets "out_channels = 2" instead of "out_channels = 10 * 3". (UNDER TEST)
	out_channels = 2, #This should be equal to quantize channels when input type is 'mulaw-quantize' else: num_distributions * 3 (prob, mean, log_scale).
	layers = 30, #Number of dilated convolutions (Default: Simplified Wavenet of Tacotron-2 paper)
	stacks = 3, #Number of dilated convolution stacks (Default: Simplified Wavenet of Tacotron-2 paper)
	residual_channels = 512,
	gate_channels = 512, #split in 2 in gated convolutions
	skip_out_channels = 256,
	kernel_size = 3,

	cin_channels = 80, #Set this to -1 to disable local conditioning, else it must be equal to num_mels!!
	upsample_conditional_features = True, #Whether to repeat conditional features or upsample them (The latter is recommended)
	upsample_scales = [15, 20], #prod(upsample_scales) should be equal to hop_size
	freq_axis_kernel_size = 3,
	leaky_alpha = 0.4,

	gin_channels = -1, #Set this to -1 to disable global conditioning, Only used for multi speaker dataset. It defines the depth of the embeddings (Recommended: 16)
	use_speaker_embedding = True, #whether to make a speaker embedding
	n_speakers = 5, #number of speakers (rows of the embedding)

	use_bias = True, #Whether to use bias in convolutional layers of the Wavenet

	max_time_sec = None,
	max_time_steps = 13000, #Max time steps in audio used to train wavenet (decrease to save memory) (Recommend: 8000 on modest GPUs, 13000 on stronger ones)
	###########################################################################################################################################

	#Tacotron Training
	tacotron_random_seed = 5339, #Determines initial graph and operations (i.e: model) random state for reproducibility
	tacotron_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

	tacotron_batch_size = 32, #number of training samples on each training steps
	tacotron_reg_weight = 1e-6, #regularization weight (for L2 regularization)
	tacotron_scale_regularization = True, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)

	tacotron_test_size = None, #% of data to keep as test data, if None, tacotron_test_batches must be not None
	tacotron_test_batches = 48, #number of test batches (For Ljspeech: 10% ~= 41 batches of 32 samples)
	tacotron_data_random_state=1234, #random state for train test split repeatability

	#Usually your GPU can handle 16x tacotron_batch_size during synthesis for the same memory amount during training (because no gradients to keep and ops to register for backprop)
	tacotron_synthesis_batch_size = 32 * 16, #This ensures GTA synthesis goes up to 40x faster than one sample at a time and uses 100% of your GPU computation power.

	tacotron_decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
	tacotron_start_decay = 50000, #Step at which learning decay starts
	tacotron_decay_steps = 50000, #Determines the learning rate decay slope (UNDER TEST)
	tacotron_decay_rate = 0.4, #learning rate decay rate (UNDER TEST)
	tacotron_initial_learning_rate = 1e-3, #starting learning rate
	tacotron_final_learning_rate = 1e-5, #minimal learning rate

	tacotron_adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
	tacotron_adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
	tacotron_adam_epsilon = 1e-6, #AdamOptimizer beta3 parameter

	tacotron_zoneout_rate = 0.1, #zoneout rate for all LSTM cells in the network
	tacotron_dropout_rate = 0.5, #dropout rate for all convolutional layers + prenet

	tacotron_clip_gradients = False, #whether to clip gradients
	natural_eval = False, #Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same teacher-forcing ratio as in training (just for overfit)

	#Decoder RNN learning can take be done in one of two ways:
	#	Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
	#	Curriculum Learning Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
	#The second approach is inspired by:
	#Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
	#Can be found under: https://arxiv.org/pdf/1506.03099.pdf
	tacotron_teacher_forcing_mode = 'constant', #Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
	tacotron_teacher_forcing_ratio = 1., #Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
	tacotron_teacher_forcing_init_ratio = 1., #initial teacher forcing ratio. Relevant if mode='scheduled'
	tacotron_teacher_forcing_final_ratio = 0., #final teacher forcing ratio. Relevant if mode='scheduled'
	tacotron_teacher_forcing_start_decay = 10000, #starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_steps = 280000, #Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_alpha = 0., #teacher forcing ratio decay rate. Relevant if mode='scheduled'
	###########################################################################################################################################

	#Wavenet Training
	wavenet_random_seed = 5339, # S=5, E=3, D=9 :)
	wavenet_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

	wavenet_batch_size = 4, #batch size used to train wavenet.
	wavenet_test_size = 0.0441, #% of data to keep as test data, if None, wavenet_test_batches must be not None
	wavenet_test_batches = None, #number of test batches.
	wavenet_data_random_state = 1234, #random state for train test split repeatability

	#During synthesis, there is no max_time_steps limitation so the model can sample much longer audio than 8k(or 13k) steps. (Audio can go up to 500k steps, equivalent to ~21sec on 24kHz)
	#Usually your GPU can handle 1x~2x wavenet_batch_size during synthesis for the same memory amount during training (because no gradients to keep and ops to register for backprop)
	wavenet_synthesis_batch_size = 4 * 2, #This ensure that wavenet synthesis goes up to 4x~8x faster when synthesizing multiple sentences. Watch out for OOM with long audios.

	wavenet_learning_rate = 1e-3,
	wavenet_adam_beta1 = 0.9,
	wavenet_adam_beta2 = 0.999,
	wavenet_adam_epsilon = 1e-8,

	wavenet_ema_decay = 0.9999, #decay rate of exponential moving average

	wavenet_dropout = 0.05, #drop rate of wavenet layers
	train_with_GTA = True, #Whether to use GTA mels to train WaveNet instead of ground truth mels.
	###########################################################################################################################################

	#Eval sentences (if no eval file was specified, these sentences are used for eval)
	sentences = [
	# From July 8, 2017 New York Times:
	'Scientists at the CERN laboratory say they have discovered a new particle.',
	'There\'s a way to measure the acute emotional intelligence that has never gone out of style.',
	'President Trump met with other leaders at the Group of 20 conference.',
	'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
	# From Google's Tacotron example page:
	'Generative adversarial network or variational auto-encoder.',
	'Basilar membrane and otolaryngology are not auto-correlations.',
	'He has read the whole thing.',
	'He reads books.',
	'He thought it was time to present the present.',
	'Thisss isrealy awhsome.',
	'Punctuation sensitivity, is working.',
	'Punctuation sensitivity is working.',
	"Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
	"She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
	"Tajima Airport serves Toyooka.",
	#From The web (random long utterance)
	'Sequence to sequence models have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization.\
	This project covers a sequence to sequence model trained to predict a speech representation from an input sequence of characters. We show that\
	the adopted architecture is able to perform this task with wild success.',
	'Thank you so much for your support!',
	]

	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
