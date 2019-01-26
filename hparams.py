import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
	# text, you may want to use "basic_cleaners" or "transliteration_cleaners".
	cleaners='english_cleaners',


	#If you only have 1 GPU or want to use only one GPU, please set num_gpus=0 and specify the GPU idx on run. example:
		#expample 1 GPU of index 2 (train on "/gpu2" only): CUDA_VISIBLE_DEVICES=2 python train.py --model='Tacotron' --hparams='tacotron_gpu_start_idx=2'
	#If you want to train on multiple GPUs, simply specify the number of GPUs available, and the idx of the first GPU to use. example:
		#example 4 GPUs starting from index 0 (train on "/gpu0"->"/gpu3"): python train.py --model='Tacotron' --hparams='tacotron_num_gpus=4, tacotron_gpu_start_idx=0'
	#The hparams arguments can be directly modified on this hparams.py file instead of being specified on run if preferred!

	#If one wants to train both Tacotron and WaveNet in parallel (provided WaveNet will be trained on True mel spectrograms), one needs to specify different GPU idxes.
	#example Tacotron+WaveNet on a machine with 4 or more GPUs. Two GPUs for each model: 
		# CUDA_VISIBLE_DEVICES=0,1 python train.py --model='Tacotron' --hparams='tacotron_num_gpus=2'
		# Cuda_VISIBLE_DEVICES=2,3 python train.py --model='WaveNet' --hparams='wavenet_num_gpus=2'

	#IMPORTANT NOTES: The Multi-GPU performance highly depends on your hardware and optimal parameters change between rigs. Default are optimized for servers.
	#If using N GPUs, please multiply the tacotron_batch_size by N below in the hparams! (tacotron_batch_size = 32 * N)
	#Never use lower batch size than 32 on a single GPU!
	#Same applies for Wavenet: wavenet_batch_size = 8 * N (wavenet_batch_size can be smaller than 8 if GPU is having OOM, minimum 2)
	#Please also apply the synthesis batch size modification likewise. (if N GPUs are used for synthesis, minimal batch size must be N, minimum of 1 sample per GPU)
	#We did not add an automatic multi-GPU batch size computation to avoid confusion in the user's mind and to provide more control to the user for
	#resources related decisions.

	#Acknowledgement:
	#	Many thanks to @MlWoo for his awesome work on multi-GPU Tacotron which showed to work a little faster than the original
	#	pipeline for a single GPU as well. Great work!

	#Hardware setup: Default supposes user has only one GPU: "/gpu:0" (Both Tacotron and WaveNet can be trained on multi-GPU: data parallelization)
	#Synthesis also uses the following hardware parameters for multi-GPU parallel synthesis.
	tacotron_num_gpus = 1, #Determines the number of gpus in use for Tacotron training.
	wavenet_num_gpus = 1, #Determines the number of gpus in use for WaveNet training.
	split_on_cpu = True, #Determines whether to split data on CPU or on first GPU. This is automatically True when more than 1 GPU is used. 
		#(Recommend: False on slow CPUs/Disks, True otherwise for small speed boost)
	###########################################################################################################################################

	#Audio
	#Audio parameters are the most important parameters to tune when using this work on your personal data. Below are the beginner steps to adapt
	#this work to your personal data:
	#	1- Determine my data sample rate: First you need to determine your audio sample_rate (how many samples are in a second of audio). This can be done using sox: "sox --i <filename>"
	#		(For this small tuto, I will consider 24kHz (24000 Hz), and defaults are 22050Hz, so there are plenty of examples to refer to)
	#	2- set sample_rate parameter to your data correct sample rate
	#	3- Fix win_size and and hop_size accordingly: (Supposing you will follow our advice: 50ms window_size, and 12.5ms frame_shift(hop_size))
	#		a- win_size = 0.05 * sample_rate. In the tuto example, 0.05 * 24000 = 1200
	#		b- hop_size = 0.25 * win_size. Also equal to 0.0125 * sample_rate. In the tuto example, 0.25 * 1200 = 0.0125 * 24000 = 300 (Can set frame_shift_ms=12.5 instead)
	#	4- Fix n_fft, num_freq and upsample_scales parameters accordingly.
	#		a- n_fft can be either equal to win_size or the first power of 2 that comes after win_size. I usually recommend using the latter
	#			to be more consistent with signal processing friends. No big difference to be seen however. For the tuto example: n_fft = 2048 = 2**11
	#		b- num_freq = (n_fft / 2) + 1. For the tuto example: num_freq = 2048 / 2 + 1 = 1024 + 1 = 1025.
	#		c- For WaveNet, upsample_scales products must be equal to hop_size. For the tuto example: upsample_scales=[15, 20] where 15 * 20 = 300
	#			it is also possible to use upsample_scales=[3, 4, 5, 5] instead. One must only keep in mind that upsample_kernel_size[0] = 2*upsample_scales[0]
	#			so the training segments should be long enough (2.8~3x upsample_scales[0] * hop_size or longer) so that the first kernel size can see the middle 
	#			of the samples efficiently. The length of WaveNet training segments is under the parameter "max_time_steps".
	#	5- Finally comes the silence trimming. This very much data dependent, so I suggest trying preprocessing (or part of it, ctrl-C to stop), then use the
	#		.ipynb provided in the repo to listen to some inverted mel/linear spectrograms. That will first give you some idea about your above parameters, and
	#		it will also give you an idea about trimming. If silences persist, try reducing trim_top_db slowly. If samples are trimmed mid words, try increasing it.
	#	6- If audio quality is too metallic or fragmented (or if linear spectrogram plots are showing black silent regions on top), then restart from step 2.
	num_mels = 80, #Number of mel-spectrogram channels and local conditioning dimensionality
	num_freq = 1025, # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
	rescale = True, #Whether to rescale audio prior to preprocessing
	rescaling_max = 0.999, #Rescaling value

	#train samples of lengths between 3sec and 14sec are more than enough to make a model capable of generating consistent speech.
	clip_mels_length = True, #For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
	max_mel_frames = 900,  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.

	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False, #Only used to set as True if using WaveNet, no difference in performance is observed in either cases.
	silence_threshold=2, #silence threshold used for sound trimming for wavenet preprocessing

	#Mel spectrogram
	n_fft = 2048, #Extra window size is filled with 0 paddings to match this parameter
	hop_size = 275, #For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
	win_size = 1100, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
	sample_rate = 22050, #22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
	frame_shift_ms = None, #Can replace hop_size parameter. (Recommended: 12.5)
	magnitude_power = 2., #The power of the spectrogram magnitude (1. for energy, 2. for power)

	#M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
	trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	trim_fft_size = 2048, #Trimming window size
	trim_hop_size = 512, #Trimmin hop length
	trim_top_db = 40, #Trimming db difference from reference db (smaller==harder trim.)

	#Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True, #Whether to normalize mel spectrograms to some predefined range (following below parameters)
	allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
	symmetric_mels = True, #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
	max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, 
																										  #not too small for fast convergence)
	normalize_for_wavenet = True, #whether to rescale to [0, 1] for wavenet. (better audio quality)
	clip_for_wavenet = True, #whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
	wavenet_pad_sides = 1, #Can be 1 or 2. 1 for pad right only, 2 for both sides padding.

	#Contribution by @begeekmyfriend
	#Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
	preemphasize = True, #whether to apply filter
	preemphasis = 0.97, #filter coefficient.

	#Limits
	min_level_db = -100,
	ref_level_db = 20,
	fmin = 55, #Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax = 7600, #To be increased/reduced depending on data.

	#Griffin Lim
	power = 1.5, #Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
	griffin_lim_iters = 60, #Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
	GL_on_GPU = True, #Whether to use G&L GPU version as part of tensorflow graph. (Usually much faster than CPU but slightly worse quality too).
	###########################################################################################################################################

	#Tacotron
	#Model general type
	outputs_per_step = 1, #number of frames to generate at each decoding step (increase to speed up computation and allows for higher batch size, decreases G&L audio quality)
	stop_at_any = True, #Determines whether the decoder should stop when predicting <stop> to any frame or to all of them (True works pretty well)
	batch_norm_position = 'after', #Can be in ('before', 'after'). Determines whether we use batch norm before or after the activation function (relu). Matter for debate.
	clip_outputs = True, #Whether to clip spectrograms to T2_output_range (even in loss computation). ie: Don't penalize model for exceeding output range and bring back to borders.
	lower_bound_decay = 0.1, #Small regularizer for noise synthesis by adding small range of penalty for silence regions. Set to 0 to clip in Tacotron range.

	#Input parameters
	embedding_dim = 512, #dimension of embedding space

	#Encoder parameters
	enc_conv_num_layers = 3, #number of encoder convolutional layers
	enc_conv_kernel_size = (5, ), #size of encoder convolution filters for each layer
	enc_conv_channels = 512, #number of encoder convolutions filters for each layer
	encoder_lstm_units = 256, #number of lstm units for each direction (forward and backward)

	#Attention mechanism
	smoothing = False, #Whether to smooth the attention normalization function
	attention_dim = 128, #dimension of attention space
	attention_filters = 32, #number of attention convolution filters
	attention_kernel = (31, ), #kernel size of attention convolution
	cumulative_weights = True, #Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

	#Attention synthesis constraints
	#"Monotonic" constraint forces the model to only look at the forwards attention_win_size steps.
	#"Window" allows the model to look at attention_win_size neighbors, both forward and backward steps.
	synthesis_constraint = False,  #Whether to use attention windows constraints in synthesis only (Useful for long utterances synthesis)
	synthesis_constraint_type = 'window', #can be in ('window', 'monotonic'). 
	attention_win_size = 7, #Side of the window. Current step does not count. If mode is window and attention_win_size is not pair, the 1 extra is provided to backward part of the window.

	#Decoder
	prenet_layers = [256, 256], #number of layers and number of units of prenet
	decoder_layers = 2, #number of decoder lstm layers
	decoder_lstm_units = 1024, #number of decoder lstm units on each layer
	max_iters = 10000, #Max decoder steps during inference (Just for safety from infinite loop cases)

	#Residual postnet
	postnet_num_layers = 5, #number of postnet convolutional layers
	postnet_kernel_size = (5, ), #size of postnet convolution filters for each layer
	postnet_channels = 512, #number of postnet convolution filters for each layer

	#CBHG mel->linear postnet
	cbhg_kernels = 8, #All kernel sizes from 1 to cbhg_kernels will be used in the convolution bank of CBHG to act as "K-grams"
	cbhg_conv_channels = 128, #Channels of the convolution bank
	cbhg_pool_size = 2, #pooling size of the CBHG
	cbhg_projection = 256, #projection channels of the CBHG (1st projection, 2nd is automatically set to num_mels)
	cbhg_projection_kernel_size = 3, #kernel_size of the CBHG projections
	cbhg_highwaynet_layers = 4, #Number of HighwayNet layers
	cbhg_highway_units = 128, #Number of units used in HighwayNet fully connected layers
	cbhg_rnn_units = 128, #Number of GRU units used in bidirectional RNN of CBHG block. CBHG output is 2x rnn_units in shape

	#Loss params
	mask_encoder = True, #whether to mask encoder padding while computing attention. Set to True for better prosody but slower convergence.
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
	#Model general type
	input_type="raw", #Raw has better quality but harder to train. mulaw-quantize is easier to train but has lower quality.
	quantize_channels=2**16,  # 65536 (16-bit) (raw) or 256 (8-bit) (mulaw or mulaw-quantize) // number of classes = 256 <=> mu = 255
	use_bias = True, #Whether to use bias in convolutional layers of the Wavenet
	legacy = True, #Whether to use legacy mode: Multiply all skip outputs but the first one with sqrt(0.5) (True for more early training stability, especially for large models)
	residual_legacy = True, #Whether to scale residual blocks outputs by a factor of sqrt(0.5) (True for input variance preservation early in training and better overall stability)

	#Model Losses parmeters
	#Minimal scales ranges for MoL and Gaussian modeling
	log_scale_min=float(np.log(1e-14)), #Mixture of logistic distributions minimal log scale
	log_scale_min_gauss = float(np.log(1e-7)), #Gaussian distribution minimal allowed log scale
	#Loss type
	cdf_loss = False, #Whether to use CDF loss in Gaussian modeling. Advantages: non-negative loss term and more training stability. (Automatically True for MoL)

	#model parameters
	#To use Gaussian distribution as output distribution instead of mixture of logistics, set "out_channels = 2" instead of "out_channels = 10 * 3". (UNDER TEST)
	out_channels = 2, #This should be equal to quantize channels when input type is 'mulaw-quantize' else: num_distributions * 3 (prob, mean, log_scale).
	layers = 20, #Number of dilated convolutions (Default: Simplified Wavenet of Tacotron-2 paper)
	stacks = 2, #Number of dilated convolution stacks (Default: Simplified Wavenet of Tacotron-2 paper)
	residual_channels = 128, #Number of residual block input/output channels.
	gate_channels = 256, #split in 2 in gated convolutions
	skip_out_channels = 128, #Number of residual block skip convolution channels.
	kernel_size = 3, #The number of inputs to consider in dilated convolutions.

	#Upsampling parameters (local conditioning)
	cin_channels = 80, #Set this to -1 to disable local conditioning, else it must be equal to num_mels!!
	#Upsample types: ('1D', '2D', 'Resize', 'SubPixel', 'NearestNeighbor')
	#All upsampling initialization/kernel_size are chosen to omit checkerboard artifacts as much as possible. (Resize is designed to omit that by nature).
	#To be specific, all initial upsample weights/biases (when NN_init=True) ensure that the upsampling layers act as a "Nearest neighbor upsample" of size "hop_size" (checkerboard free).
	#1D spans all frequency bands for each frame (channel-wise) while 2D spans "freq_axis_kernel_size" bands at a time. Both are vanilla transpose convolutions.
	#Resize is a 2D convolution that follows a Nearest Neighbor (NN) resize. For reference, this is: "NN resize->convolution".
	#SubPixel (2D) is the ICNR version (initialized to be equivalent to "convolution->NN resize") of Sub-Pixel convolutions. also called "checkered artifact free sub-pixel conv".
	#Finally, NearestNeighbor is a non-trainable upsampling layer that just expands each frame (or "pixel") to the equivalent hop size. Ignores all upsampling parameters.
	upsample_type = 'SubPixel', #Type of the upsampling deconvolution. Can be ('1D' or '2D', 'Resize', 'SubPixel' or simple 'NearestNeighbor').
	upsample_activation = 'Relu', #Activation function used during upsampling. Can be ('LeakyRelu', 'Relu' or None)
	upsample_scales = [11, 25], #prod(upsample_scales) should be equal to hop_size
	freq_axis_kernel_size = 3, #Only used for 2D upsampling types. This is the number of requency bands that are spanned at a time for each frame.
	leaky_alpha = 0.4, #slope of the negative portion of LeakyRelu (LeakyRelu: y=x if x>0 else y=alpha * x)
	NN_init = True, #Determines whether we want to initialize upsampling kernels/biases in a way to ensure upsample is initialize to Nearest neighbor upsampling. (Mostly for debug)
	NN_scaler = 0.3, #Determines the initial Nearest Neighbor upsample values scale. i.e: upscaled_input_values = input_values * NN_scaler (1. to disable)

	#global conditioning
	gin_channels = -1, #Set this to -1 to disable global conditioning, Only used for multi speaker dataset. It defines the depth of the embeddings (Recommended: 16)
	use_speaker_embedding = True, #whether to make a speaker embedding
	n_speakers = 5, #number of speakers (rows of the embedding)
	speakers_path = None, #Defines path to speakers metadata. Can be either in "speaker\tglobal_id" (with header) tsv format, or a single column tsv with speaker names. If None, use "speakers".
	speakers = ['speaker0', 'speaker1', #List of speakers used for embeddings visualization. (Consult "wavenet_vocoder/train.py" if you want to modify the speaker names source).
				'speaker2', 'speaker3', 'speaker4'], #Must be consistent with speaker ids specified for global conditioning for correct visualization.
	###########################################################################################################################################

	#Tacotron Training
	#Reproduction seeds
	tacotron_random_seed = 5339, #Determines initial graph and operations (i.e: model) random state for reproducibility
	tacotron_data_random_state = 1234, #random state for train test split repeatability

	#performance parameters
	tacotron_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

	#train/test split ratios, mini-batches sizes
	tacotron_batch_size = 32, #number of training samples on each training steps
	#Tacotron Batch synthesis supports ~16x the training batch size (no gradients during testing). 
	#Training Tacotron with unmasked paddings makes it aware of them, which makes synthesis times different from training. We thus recommend masking the encoder.
	tacotron_synthesis_batch_size = 1, #DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN'T TRAIN TACOTRON WITH "mask_encoder=True"!!
	tacotron_test_size = 0.05, #% of data to keep as test data, if None, tacotron_test_batches must be not None. (5% is enough to have a good idea about overfit)
	tacotron_test_batches = None, #number of test batches.

	#Learning rate schedule
	tacotron_decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
	tacotron_start_decay = 40000, #Step at which learning decay starts
	tacotron_decay_steps = 18000, #Determines the learning rate decay slope (UNDER TEST)
	tacotron_decay_rate = 0.5, #learning rate decay rate (UNDER TEST)
	tacotron_initial_learning_rate = 1e-3, #starting learning rate
	tacotron_final_learning_rate = 1e-4, #minimal learning rate

	#Optimization parameters
	tacotron_adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
	tacotron_adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
	tacotron_adam_epsilon = 1e-6, #AdamOptimizer Epsilon parameter

	#Regularization parameters
	tacotron_reg_weight = 1e-6, #regularization weight (for L2 regularization)
	tacotron_scale_regularization = False, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)
	tacotron_zoneout_rate = 0.1, #zoneout rate for all LSTM cells in the network
	tacotron_dropout_rate = 0.5, #dropout rate for all convolutional layers + prenet
	tacotron_clip_gradients = True, #whether to clip gradients

	#Evaluation parameters
	tacotron_natural_eval = False, #Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same teacher-forcing ratio as in training (just for overfit)

	#Decoder RNN learning can take be done in one of two ways:
	#	Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
	#	Scheduled Sampling Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
	#The second approach is inspired by:
	#Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
	#Can be found under: https://arxiv.org/pdf/1506.03099.pdf
	tacotron_teacher_forcing_mode = 'constant', #Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
	tacotron_teacher_forcing_ratio = 1., #Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
	tacotron_teacher_forcing_init_ratio = 1., #initial teacher forcing ratio. Relevant if mode='scheduled'
	tacotron_teacher_forcing_final_ratio = 0., #final teacher forcing ratio. (Set None to use alpha instead) Relevant if mode='scheduled'
	tacotron_teacher_forcing_start_decay = 10000, #starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_steps = 40000, #Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_alpha = None, #teacher forcing ratio decay rate. Defines the final tfr as a ratio of initial tfr. Relevant if mode='scheduled'

	#Speaker adaptation parameters
	tacotron_fine_tuning = False, #Set to True to freeze encoder and only keep training pretrained decoder. Used for speaker adaptation with small data.
	###########################################################################################################################################

	#Wavenet Training
	wavenet_random_seed = 5339, # S=5, E=3, D=9 :)
	wavenet_data_random_state = 1234, #random state for train test split repeatability

	#performance parameters
	wavenet_swap_with_cpu = False, #Whether to use cpu as support to gpu for synthesis computation (while loop).(Not recommended: may cause major slowdowns! Only use when critical!)

	#train/test split ratios, mini-batches sizes
	wavenet_batch_size = 8, #batch size used to train wavenet.
	#During synthesis, there is no max_time_steps limitation so the model can sample much longer audio than 8k(or 13k) steps. (Audio can go up to 500k steps, equivalent to ~21sec on 24kHz)
	#Usually your GPU can handle ~2x wavenet_batch_size during synthesis for the same memory amount during training (because no gradients to keep and ops to register for backprop)
	wavenet_synthesis_batch_size = 10 * 2, #This ensure that wavenet synthesis goes up to 4x~8x faster when synthesizing multiple sentences. Watch out for OOM with long audios.
	wavenet_test_size = None, #% of data to keep as test data, if None, wavenet_test_batches must be not None
	wavenet_test_batches = 1, #number of test batches.

	#Learning rate schedule
	wavenet_lr_schedule = 'exponential', #learning rate schedule. Can be ('exponential', 'noam')
	wavenet_learning_rate = 1e-3, #wavenet initial learning rate
	wavenet_warmup = float(4000), #Only used with 'noam' scheme. Defines the number of ascending learning rate steps.
	wavenet_decay_rate = 0.5, #Only used with 'exponential' scheme. Defines the decay rate.
	wavenet_decay_steps = 200000, #Only used with 'exponential' scheme. Defines the decay steps.

	#Optimization parameters
	wavenet_adam_beta1 = 0.9, #Adam beta1
	wavenet_adam_beta2 = 0.999, #Adam beta2
	wavenet_adam_epsilon = 1e-6, #Adam Epsilon

	#Regularization parameters
	wavenet_clip_gradients = True, #Whether the clip the gradients during wavenet training.
	wavenet_ema_decay = 0.9999, #decay rate of exponential moving average
	wavenet_weight_normalization = False, #Whether to Apply Saliman & Kingma Weight Normalization (reparametrization) technique. (Used in DeepVoice3, not critical here)
	wavenet_init_scale = 1., #Only relevent if weight_normalization=True. Defines the initial scale in data dependent initialization of parameters.
	wavenet_dropout = 0.05, #drop rate of wavenet layers
	wavenet_gradient_max_norm = 100.0, #Norm used to clip wavenet gradients
	wavenet_gradient_max_value = 5.0, #Value used to clip wavenet gradients

	#training samples length
	max_time_sec = None, #Max time of audio for training. If None, we use max_time_steps.
	max_time_steps = 11000, #Max time steps in audio used to train wavenet (decrease to save memory) (Recommend: 8000 on modest GPUs, 13000 on stronger ones)

	#Evaluation parameters
	wavenet_natural_eval = False, #Whether to use 100% natural eval (to evaluate autoregressivity performance) or with teacher forcing to evaluate overfit and model consistency.

	#Tacotron-2 integration parameters
	train_with_GTA = True, #Whether to use GTA mels to train WaveNet instead of ground truth mels.
	###########################################################################################################################################

	#Eval/Debug parameters
	#Eval sentences (if no eval text file was specified during synthesis, these sentences are used for eval)
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
	'The big brown fox jumps over the lazy dog.',
	'Did the big brown fox jump over the lazy dog?',
	"Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
	"She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
	"Tajima Airport serves Toyooka.",
	#From The web (random long utterance)
	# 'On offering to help the blind man, the man who then stole his car, had not, at that precise moment, had any evil intention, quite the contrary, \
	# what he did was nothing more than obey those feelings of generosity and altruism which, as everyone knows, \
	# are the two best traits of human nature and to be found in much more hardened criminals than this one, a simple car-thief without any hope of advancing in his profession, \
	# exploited by the real owners of this enterprise, for it is they who take advantage of the needs of the poor.',
	# A final Thank you note!
	'Thank you so much for your support!',
	],

	#Wavenet Debug
	wavenet_synth_debug = False, #Set True to use target as debug in WaveNet synthesis. 
	wavenet_debug_wavs = ['training_data/audio/audio-LJ001-0008.npy'], #Path to debug audios. Must be multiple of wavenet_num_gpus.
	wavenet_debug_mels = ['training_data/mels/mel-LJ001-0008.npy'], #Path to corresponding mels. Must be of same length and order as wavenet_debug_wavs.

	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
