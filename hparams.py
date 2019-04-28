import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
	# text, you may want to use "basic_cleaners" or "transliteration_cleaners".
	cleaners='basic_cleaners',

	#Hardware setup (TODO: multi-GPU parallel tacotron training)
	use_all_gpus = False, #Whether to use all GPU resources. If True, total number of available gpus will override num_gpus.
	num_gpus = 1, #Determines the number of gpus in use
	###########################################################################################################################################

	#Audio
	num_mels = 160, #Number of mel-spectrogram channels and local conditioning dimensionality
	num_freq = 2049, # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
	rescale = False, #Whether to rescale audio prior to preprocessing
	rescaling_max = 0.999, #Rescaling value
	trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	clip_mels_length = True, #For cases of OOM (Not really recommended, working on a workaround)
	max_mel_frames = 600,  #Only relevant when clip_mels_length = True
	max_text_length = 150,  #Only relevant when clip_mels_length = True

	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False,
	silence_threshold=2, #silence threshold used for sound trimming for wavenet preprocessing

	#Mel spectrogram
	n_fft = 4096, #Extra window size is filled with 0 paddings to match this parameter
	hop_size = 600, #For 22050Hz, 275 ~= 12.5 ms
	win_size = 2400, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft)
	sample_rate = 48000, #22050 Hz (corresponding to ljspeech dataset)
	frame_shift_ms = None,
	preemphasis = 0.97, # preemphasis coefficient

	#M-AILABS (and other datasets) trim params
	trim_fft_size = 512,
	trim_hop_size = 128,
	trim_top_db = 60,

	#Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True,
	allow_clipping_in_normalization = False, #Only relevant if mel_normalization = True
	symmetric_mels = True, #Whether to scale the data to be symmetric around 0
	max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max]
	normalize_for_wavenet = True, #whether to rescale to [0, 1] for wavenet.

	#Limits
	min_level_db = -120,
	ref_level_db = 20,
	fmin = 125, #Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
	fmax = 7600,

	#Griffin Lim
	power = 1.2,
	griffin_lim_iters = 60,
	###########################################################################################################################################

	#Tacotron
	outputs_per_step = 2, #number of frames to generate at each decoding step (speeds up computation and allows for higher batch size)
	stop_at_any = False, #Determines whether the decoder should stop when predicting <stop> to any frame or to all of them
	batch_norm_position = 'after', #Can be in ('before', 'after'). Determines whether we use batch norm before or after the activation function (relu). Matter for debate.

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

	#Attention synthesis constraints
	#"Monotonic" constraint forces the model to only look at the forwards attention_win_size steps.
	#"Window" allows the model to look at attention_win_size neighbors, both forward and backward steps.
	synthesis_constraint = False,  #Whether to use attention windows constraints in synthesis only (Useful for long utterances synthesis)
	synthesis_constraint_type = 'window', #can be in ('window', 'monotonic').
	attention_win_size = 7, #Side of the window. Current step does not count. If mode is window and attention_win_size is not pair, the 1 extra is provided to backward part of the window.

	prenet_layers = [256, 256], #number of layers and number of units of prenet
	decoder_layers = 2, #number of decoder lstm layers
	decoder_lstm_units = 1024, #number of decoder lstm units on each layer
	max_iters = 1000, #Max decoder steps during inference (Just for safety from infinite loop cases)

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
	mask_encoder = False, #whether to mask encoder padding while computing attention. Set to True for better prosody but slower convergence.
	mask_decoder = False, #Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)
	cross_entropy_pos_weight = 1, #Use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1 = disabled)
	predict_linear = True, #Whether to add a post-processing network to the Tacotron to predict linear spectrograms (True mode Not tested!!)
	###########################################################################################################################################

	#Tacotron Training
	#Reproduction seeds
	tacotron_random_seed = 5339, #Determines initial graph and operations (i.e: model) random state for reproducibility
	tacotron_data_random_state = 1234, #random state for train test split repeatability

	#performance parameters
	tacotron_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

	#train/test split ratios, mini-batches sizes
	tacotron_batch_size = 48, #number of training samples on each training steps
	#Tacotron Batch synthesis supports ~16x the training batch size (no gradients during testing).
	#Training Tacotron with unmasked paddings makes it aware of them, which makes synthesis times different from training. We thus recommend masking the encoder.
	tacotron_synthesis_batch_size = 1, #DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN'T TRAIN TACOTRON WITH "mask_encoder=True"!!
	tacotron_test_size = 0.05, #% of data to keep as test data, if None, tacotron_test_batches must be not None. (5% is enough to have a good idea about overfit)
	tacotron_test_batches = None, #number of test batches.

	#Learning rate schedule
	tacotron_decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
	tacotron_start_decay = 40000, #Step at which learning decay starts
	tacotron_decay_steps = 40000, #Determines the learning rate decay slope (UNDER TEST)
	tacotron_decay_rate = 0.4, #learning rate decay rate (UNDER TEST)
	tacotron_initial_learning_rate = 1e-3, #starting learning rate
	tacotron_final_learning_rate = 1e-5, #minimal learning rate

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
	#       Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
	#       Scheduled Sampling Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
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
	###########################################################################################################################################

	#Eval sentences (if no eval file was specified, these sentences are used for eval)
	sentences = [
	# "huan2 qiu2 wang3 bao4 dao4 .",
	# "e2 luo2 si1 wei4 xing1 wang3 shi2 yi1 ri4 bao4 dao4 cheng1 .",
	# "ji4 nian4 di4 yi2 ci4 shi4 jie4 da4 zhan4 jie2 shu4 yi4 bai3 zhou1 nian2 qing4 zhu4 dian3 li3 zai4 ba1 li2 ju3 xing2 .",
	# "e2 luo2 si1 zong3 tong3 pu3 jing1 he2 mei3 guo2 zong3 tong3 te4 lang3 pu3 ,",
	# "zai4 ba1 li2 kai3 xuan2 men2 jian4 mian4 shi2 wo4 shou3 zhi4 yi4 .",
	# "pu3 jing1 biao3 shi4 .",
	# "tong2 mei3 guo2 zong3 tong3 te4 lang3 pu3 ,",
	# "jin4 xing2 le hen3 hao3 de jiao1 liu2 .",
	# "e2 luo2 si1 zong3 tong3 zhu4 shou3 you2 li3 wu1 sha1 ke1 fu1 biao3 shi4 .",
	# "fa3 guo2 fang1 mian4 zhi2 yi4 yao1 qiu2 ,",
	# "bu2 yao4 zai4 ba1 li2 ju3 xing2 ji4 nian4 huo2 dong4 qi1 jian1 ." ,
	# "ju3 xing2 e2 mei3 liang3 guo2 zong3 tong3 de dan1 du2 hui4 wu4 da2 cheng2 le xie2 yi4 .",
	# "wo3 men yi3 jing1 kai1 shi3 xie2 tiao2 ,",
	# "e2 luo2 si1 he2 mei3 guo2 zong3 tong3 hui4 wu4 de shi2 jian1 .",
	# "dan4 hou4 lai2 ,",
	# "wo3 men kao3 lv4 dao4 le fa3 guo2 tong2 hang2 men de dan1 you1 he2 guan1 qie4 .",
	# "wu1 sha1 ke1 fu1 shuo1 .",
	# "yin1 ci3 , wo3 men yu3 mei3 guo2 dai4 biao3 men yi4 qi3 jin4 xing2 le tao3 lun4 .",
	# "jue2 ding4 zai4 bu4 yi2 nuo4 si1 ai4 li4 si1 feng1 hui4 shang4 ,",
	# "jin4 xing2 nei4 rong2 geng4 feng1 fu4 de dui4 hua4 .",
	# "bao4 dao4 cheng1 .",
	# "pu3 jing1 he2 te4 lang3 pu3 zai4 ai4 li4 she4 gong1 wu3 can1 hui4 shang4 de zuo4 wei4 an1 pai2 ,",
	# "zai4 zui4 hou4 yi4 fen1 zhong1 jin4 xing2 le tiao2 zheng3 .",
	# "dan4 zhe4 bing4 bu4 fang2 ai4 ta1 men jiao1 tan2 .",
	# "sui1 ran2 dong1 dao4 zhu3 fa3 guo2 dui4 ta1 men zai4 ba1 li2 de hui4 wu4 biao3 shi4 fan3 dui4 .",
	# "dan4 e2 mei3 ling3 dao3 ren2 reng2 ran2 biao3 shi4 .",
	# "ta1 men xi1 wang4 zai4 ai4 li4 she4 gong1 de gong1 zuo4 wu3 can1 shang4 hui4 mian4 .",
	# "chu1 bu4 zuo4 wei4 biao3 xian3 shi4 .",
	# "te4 lang3 pu3 bei4 an1 pai2 zai4 pu3 jing1 pang2 bian1 .",
	# "dan4 zai4 sui2 hou4 jin4 xing2 de gong1 zuo4 wu3 can1 qi1 jian1 .",
	# "zuo4 wei4 an1 pai2 xian3 ran2 yi3 jing1 fa1 sheng1 le bian4 hua4 .",
	# "cong2 zhao4 pian1 lai2 kan4 .",
	# "pu3 jing1 dang1 shi2 zheng4 quan2 shen2 guan4 zhu4 de yu3 lian2 he2 guo2 mi4 shu1 zhang3 gu3 te4 lei2 si1 jiao1 tan2 .",
	# "ou1 meng2 wei3 yuan2 hui4 zhu3 xi2 rong2 ke4 zuo4 zai4 pu3 jing1 de you4 bian1 .",
	# "er2 te4 lang3 pu3 ze2 zuo4 zai4 ma3 ke4 long2 pang2 bian1 .",
	# "ma3 ke4 long2 de you4 bian1 ze2 shi4 de2 guo2 zong3 li3 mo4 ke4 er3 .",
	# "ci3 qian2 . pu3 jing1 zai4 fang3 wen4 ba1 li2 qi1 jian1 biao3 shi4 .",
	# "ta1 bu4 pai2 chu2 yu3 te4 lang3 pu3 zai4 gong1 zuo4 wu3 can1 shi2 jin4 xing2 jiao1 liu2 .",
	# "pu3 jing1 zai4 fa3 guo2 pin2 dao4 de jie2 mu4 zhong1 hui2 da2 ,",
	# "shi4 fou3 yi3 tong2 te4 lang3 pu3 jin4 xing2 jiao1 liu2 de wen4 ti2 shi2 biao3 shi4 ,",
	# "zan4 shi2 mei2 you3 .",
	# "wo3 men zhi3 da3 le ge4 zhao1 hu .",
	# "yi2 shi4 yi3 zhe4 yang4 de fang1 shi4 jin4 xing2 .",
	# "wo3 men wu2 fa3 zai4 na4 li3 jin4 xing2 jiao1 liu2 .",
	# "wo3 men guan1 kan4 le fa1 sheng1 de shi4 qing2 .",
	# "dan4 xian4 zai4 hui4 you3 gong1 zuo4 wu3 can1 .",
	# "ye3 xu3 zai4 na4 li3 .",
	# "wo3 men hui4 jin4 xing2 jie1 chu4 .",
	# "dan4 shi4 . wu2 lun4 ru2 he2 .",
	# "wo3 men shang1 ding4 .",
	# "wo3 men zai4 zhe4 li3 ,",
	# "bu2 hui4 wei2 fan3 zhu3 ban4 guo2 de gong1 zuo4 an1 pai2 .",
	# "gen1 ju4 ta1 men de yao1 qiu2 .",
	# "wo3 men bu2 hui4 zai4 zhe4 li3 zu3 zhi1 ren4 he2 hui4 mian4 .",
	# "er2 shi4 ke3 neng2 hui4 zai4 feng1 hui4 qi1 jian1 ,",
	# "huo4 zai4 ci3 zhi1 hou4 ju3 xing2 hui4 mian4 .",
	# "pu3 jing1 hai2 biao3 shi4 .",
	# "e2 luo2 si1 zhun3 bei4 tong2 mei3 guo2 jin4 xing2 dui4 hua4 .",
	# "fan3 zheng4 bu2 shi4 mo4 si1 ke1 yao4 tui4 chu1 zhong1 dao3 tiao2 yue1 .",

	# "guan1 yu2 xi1 zang4 de chuan2 shuo1 you3 hen3 duo1 ,",
	# "li4 lai2 , dou1 shi4 chao2 sheng4 zhe3 de tian1 tang2 ,",
	# "er2 zuo4 wei2 zhong1 guo2 xi1 nan2 bian1 chui2 zhong4 de4 ,",
	# "ye3 dou1 shi4 zhong1 guo2 ling3 tu3 bu4 ke3 fen1 ge1 de yi2 bu4 fen .",
	# "er4 ling2 yi1 wu3 nian2 , yang1 shi4 ceng2 jing1 bo1 chu1 guo4 yi2 bu4 gao1 fen1 ji4 lu4 pian4 ,",
	# "di4 san1 ji2",
	# "pian4 zhong1 , tian1 gao1 di4 kuo4 de feng1 jing3 ,",
	# "rang4 wu2 shu4 ren2 dui4 xi1 zang4 qing2 gen1 shen1 zhong4 ."
	# "shi2 ge2 liang3 nian2 , you2 yuan2 ban1 ren2 ma3 da3 zao4 de jie3 mei4 pian1 ,"
	# "ji2 di4 , qiao1 ran2 shang4 xian4 !",
	# "mei3 yi4 zheng4 dou1 shi4 bi4 zhi3 , mei3 yi2 mu4 dou1 shi4 ren2 jian1 xian1 jing4 .",
	# "zi4 ying3 pian1 bo1 chu1 zhi1 lai2 , hao3 ping2 ru2 chao2 ,",
	# "jiu4 lian2 yi2 xiang4 yi3 yan2 jin3 chu1 ming2 de dou4 ban4 ping2 fen1 ye3 shi4 hen3 gao1 .",
	# "zao3 zai4 er4 ling2 yi1 wu3 nian2 ,",
	# "ta1 de di4 yi1 ji4 di4 san1 ji2 jiu4 na2 dao4 le dou4 ban4 jiu2 dian3 er4 fen1 .",
	# "er2 rang4 ta1 yi2 xia4 na2 dao4 jiu2 dian3 wu3 fen1 de yuan2 yin1 shi4 yin1 wei4, ",
	# "ta1 zhan3 shi4 le zai4 na4 pian4 jue2 mei3 yu3 pin2 ji2 bing4 cun2 de jing4 tu3 shang4 ,",
	# "pu3 tong1 ren2 de zhen1 shi2 sheng1 huo2 shi4 shen2 me yang4 zi .",

	"bai2 jia1 xuan1 hou4 lai2 yin2 yi3 hao2 zhuang4 de shi4 yi4 sheng1 li3 qu3 guo4 qi1 fang2 nv3 ren2 .",
	"qu3 tou2 fang2 xi2 fu4 shi2 ta1 gang1 gang1 guo4 shi2 liu4 sui4 sheng1 ri4 .",
	"na4 shi4 xi1 yuan2 shang4 gong3 jia1 cun1 da4 hu4 gong3 zeng1 rong2 de tou2 sheng1 nv3 ,",
	"bi3 ta1 da4 liang3 sui4 .",
	"ta1 zai4 wan2 quan2 wu2 zhi1 huang1 luan4 zhong1 , du4 guo4 le xin1 hun1 zhi1 ye4 ,",
	"liu2 xia4 le yong2 yuan3 xiu1 yu2 xiang4 ren2 dao4 ji2 de ke3 xiao4 de sha3 yang4 ,",
	"er2 zi4 ji3 que4 yong3 sheng1 nan2 yi3 wang4 ji4 .",
	"yi4 nian2 hou4 , zhe4 ge4 nv3 ren2 si3 yu2 nan2 chan3 .",
	"di4 er4 fang2 qu3 de shi4 nan2 yuan2 pang2 jia1 cun1 yin1 shi2 ren2 jia1 , pang2 xiu1 rui4 de nai3 gan1 nv3 er2 .",
	"zhe4 nv3 zi3 you4 zheng4 hao3 bi3 ta1 xiao2 liang3 sui4 ,",
	"mu2 yang4 jun4 xiu4 yan3 jing1 hu1 ling2 er .",
	"ta1 wan2 quan2 bu4 zhi1 dao4 jia4 ren2 shi4 zen3 me hui2 shi4 ,",
	"er2 ta1 ci3 shi2 yi3 an1 shu2 nan2 nv3 zhi1 jian1 suo2 you3 de yin3 mi4 .",
	"ta1 kan4 zhe ta1 de xiu1 qie4 huang1 luan4 er2 xiang3 dao4 zi4 ji3 di4 yi1 ci4 de sha3 yang4 fan3 dao4 jue2 de geng4 fu4 ci4 ji .",
	"dang1 ta1 hong1 suo1 zhe ba3 duo2 duo3 shan2 shan3 er2 you4 bu4 gan3 wei2 ao4 ta1 de xiao3 xi2 fu4 guo3 ru4 shen1 xia4 de shi2 hou4 ,",
	"ta1 ting1 dao4 le ta1 de bu2 shi4 huan1 le4 er2 shi4 tong4 ku3 de yi4 sheng1 ku1 jiao4 .",
	"dang1 ta1 pi2 bei4 de xie1 xi1 xia4 lai2 ,",
	"cai2 fa1 jue2 jian1 bang3 nei4 ce4 teng2 tong4 zuan1 xin1 ,",
	"ta1 ba3 ta1 yao3 lan4 le .",
	"ta1 fu3 shang1 xi1 tong4 de shi2 hou4 ,",
	"xin1 li3 jiu4 chao2 qi3 le dui4 zhe4 ge4 jiao1 guan4 de you2 dian3 ren4 xing4 de nai3 gan1 nv3 er de nao2 huo3 .",
	"zheng4 yu4 fa1 zuo4 ,",
	"ta1 que4 ban1 guo4 ta1 de jian1 bang3 an4 shi4 ta1 zai4 lai2 yi1 ci4 .",
	"yi4 dang1 jing1 guo4 nan2 nv3 jian1 de di4 yi1 ci4 jiao1 huan1 ,",
	"ta1 jiu4 bian4 de2 mei2 you3 jie2 zhi4 de ren4 xing4 .",
	"zhe4 ge4 nv3 ren2 cong2 xia4 jiao4 ding3 zhe hong2 chou2 gai4 jin1 , jin4 ru4 bai2 jia1 men2 lou2 ,",
	"dao4 tang3 jin4 yi2 ju4 bao2 ban3 guan1 cai tai2 chu1 zhe4 ge4 men2 lou2 ,",
	"shi2 jian1 shang4 bu4 zu2 yi1 nian2 , shi4 hai4 lao2 bing4 si3 de .",
	]

	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
