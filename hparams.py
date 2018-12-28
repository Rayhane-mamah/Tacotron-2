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
	rescale = True, #Whether to rescale audio prior to preprocessing
	rescaling_max = 0.999, #Rescaling value
	trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	clip_mels_length = True, #For cases of OOM (Not really recommended, working on a workaround)
	max_mel_frames = 900,  #Only relevant when clip_mels_length = True

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

	#Tacotron Training
	tacotron_random_seed = 5339, #Determines initial graph and operations (i.e: model) random state for reproducibility
	tacotron_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

	tacotron_batch_size = 32, #number of training samples on each training steps
	tacotron_reg_weight = 1e-7, #regularization weight (for L2 regularization)
	tacotron_scale_regularization = True, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)

	tacotron_test_size = None, #% of data to keep as test data, if None, tacotron_test_batches must be not None
	tacotron_test_batches = 32, #number of test batches (For Ljspeech: 10% ~= 41 batches of 32 samples)
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

	tacotron_clip_gradients = True, #whether to clip gradients
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
	tacotron_teacher_forcing_start_decay = 40000, #starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_steps = 280000, #Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_alpha = 0., #teacher forcing ratio decay rate. Relevant if mode='scheduled'
	###########################################################################################################################################

	#Eval sentences (if no eval file was specified, these sentences are used for eval)
	sentences = [
	# "yu2 jian4 jun1 : wei4 mei3 ge4 you3 cai2 neng2 de ren2 ti2 gong1 ping2 tai2 .",
	# "ta1 shi4 yin1 pin2 ling3 yu4 de tao2 bao3 tian1 mao1 , zai4 zhe4 ge4 ping2 tai2 shang4, ",
	# "mei3 ge4 nei4 rong2 sheng1 chan3 zhe3 dou1 ke3 yi3 hen3 fang1 bian4 de shi1 xian4 zi4 wo3 jia4 zhi2 , geng4 duo1 de ren2 yong1 you3 wei1 chuang4 ye4 de ji1 hui4 .",
	# "zui4 jin4 xi3 ma3 la1 ya3 de bao4 guang1 lv4 you3 dian3 gao1 , ren4 xing4 shai4 chu1 yi1 dian3 qi1 yi4 yuan2 de zhang4 hu4 yu2 e2 de jie2 tu2 ,",
	# "rang4 ye4 nei4 ye4 wai4 dou1 hen3 jing1 tan4 : yi2 ge4 zuo4 yin1 pin2 de , ju1 ran2 you3 zhe4 me duo1 qian2 ?",
	# "ji4 zhe3 cha2 dao4 , wang3 shang4 dui4 xi3 ma3 la1 ya3 de jie4 shao4 shi4 ,",
	# "xun4 su4 cheng2 zhang3 wei4 zhong1 guo2 zui4 da4 de yin1 pin2 fen1 xiang3 ping2 tai2 , mu4 qian2 yi3 yong1 you3 liang3 yi4 yong4 hu4 , qi3 ye4 zong3 gu1 zhi2 chao1 guo4 san1 shi2 yi4 yuan2 ren2 min2 bi4 .",
	# "jin4 ri4 , ji4 zhe3 zai4 shang4 hai3 zhang1 jiang1 gao1 ke1 ji4 yuan2 qu1 de xi3 ma3 la1 ya3 ji1 di4 zhuan1 fang3 le yu2 jian4 jun1 .",
	# "ta1 men dou1 shi4 han3 ta1 lao3 yu2 de , bu4 guo4 hou4 lai2 ji4 zhe3 wen4 guo4 ta1 de nian2 ling2 , qi2 shi2 cai2 yi1 jiu3 qi1 qi1 nian2 de .",
	# "ji4 zhe3 liao3 jie3 dao4 , xi3 ma3 la1 ya3 cai3 qu3 bu4 duo1 jian4 de lian2 xi2 mo2 shi4 , ling4 yi1 wei4 jiu4 shi4 chen2 xiao3 yu3 ,",
	# "liang3 ren2 qi4 zhi4 hun4 da1 , you3 dian3 nan2 zhu3 wai4 nv3 zhu3 nei4 de yi4 si1 ,",
	# "bu4 guo4 ta1 men zhi3 shi4 da1 dang4 , bu2 shi4 chang2 jian4 de fu1 qi1 dang4 mo2 shi4 . yong4 yu2 jian4 jun1 de hua4 lai2 shuo1 , zhe4 ge4 mo2 shi4 ye3 bu4 chang2 jian4 .",

	# bei3 hai3 dao4 deng3 di4 you3 qiang2 lie4 zhen4 gan3 ri4 ben3 qi4 xiang4 ting1 shuo1 , di4 zhen4 fa1 sheng1 yu2 dang1 di4 shi2 jian1 liu4 ri4 ling2 chen2 san1 shi2 ba1 fen1 bei3 jing1 shi2 jian1 er4 shi2 ba1 fen1 ",
	# zhen4 zhong1 wei4 yu2 bei3 hai3 dao4 de dan3 zhen4 di4 qu1 zhong1 dong1 bu4 , zhen4 yuan2 shen1 du4 yue1 si4 shi2 gong1 li3 . bei3 hai3 dao4 , qing1 sen1 xian4 , yan2 shou3 xian4 , gong1 cheng2 xian4 deng3 di4 you3 qiang2 lie4 zhen4 gan3 .",
	# qi4 xiang4 ting1 shuo1 , ci3 ci4 di4 zhen4 ke3 neng2 yin3 fa1 zhen4 zhong1 fu4 jin4 di4 qu1 hai3 mian4 chao2 wei4 bo1 dong4 , dan4 mei2 you3 yin3 fa1 hai3 xiao4 de feng1 xian3 .",
	# ju4 zhong1 guo2 di4 zhen4 tai2 wang3 xiao1 xi1 , dang1 di4 shi2 jian1 liu4 hao4 ling2 chen2 , bei3 hai3 dao4 fa1 sheng1 li3 shi4 liu4 dian3 jiu3 ji2 di4 zhen4 , zhen4 yuan2 shen1 du4 si4 shi2 gong1 li3 .",
	# ju4 ri4 ben3 mei2 ti3 de bao4 dao4 , di4 zhen4 mu4 qian2 yi3 jing1 zao4 cheng2 zha2 huang3 deng3 duo1 shu4 di4 qu1 ting2 dian4 , bei3 hai3 dao4 dian4 li4 gong1 si1 cheng1 , di4 zhen4 zao4 cheng2 le yue1 wu3 bai3 er4 shi2 wu3 wan4 hu4 ju1 min2 jia1 zhong1 ting2 dian4 .",
	# yi1 xie1 dao4 lu4 yin1 di4 zhen4 shou4 sun3 huo4 chu1 xian4 duan4 lie4 di2 qing2 kuang4 , yi1 xie1 di4 fang1 hai2 fa1 sheng1 le huo3 zai1 , yi1 bu4 fen4 gao1 su4 gong1 lu4 ye3 yi3 jing1 feng1 xing2 .",
	# bei3 hai3 dao4 po1 he2 dian4 zhan4 yin1 wei4 ting2 dian4 , wai4 bu4 gong1 dian4 zhong1 duan4 , mu4 qian2 shi3 yong4 jin3 ji2 fa1 dian4 she4 bei4 gong1 dian4 , he2 ran2 liao4 leng3 que4 zhuang1 zhi4 zheng4 chang2 yun4 zhuan3 .",
	# ling4 ju4 ri4 mei2 bao4 dao4 , ben3 ci4 di4 zhen4 qiang2 lie4 yu2 zhen4 reng2 zai4 ji4 xu4 . di4 zhen4 yi3 zao4 cheng2 duo1 ren2 shou4 shang1 , bu4 fen4 jian4 zhu4 sun3 hui3 , dang1 di4 xue2 xiao4 jin1 ri4 ting2 ke4 , tie3 lu4 fu2 wu4 zan4 ting2 .",
	# bei3 hai3 dao4 zhu3 yao4 tie3 lu4 yun4 ying2 shang1 biao3 shi4 , yi3 zan4 ting2 bao1 kuo4 xin1 gan4 xian4 zi3 dan4 tou2 lie4 che1 zai4 nei4 de suo3 you3 fu2 wu4 . gai1 gong1 si1 biao3 shi4 , shang4 bu4 que4 ding4 fu2 wu4 he2 shi2 hui1 fu4 .",
	# ju4 ri4 ben3 guo2 tu3 jiao1 tong1 sheng3 xin1 qian1 sui4 ji1 chang3 shi4 wu4 suo3 xiao1 xi1 , bei3 hai3 dao4 de xin1 qian1 sui4 ji1 chang3 hou4 ji1 lou2 duo1 chu4 qiang2 bi4 shou4 sun3 ,",
	# wei4 que4 ren4 an1 quan2 jiang1 guan1 bi4 ji1 chang3 da4 lou2 , mu4 qian2 wu2 fa3 que4 ren4 hang2 ban1 ke3 yi3 zheng4 chang2 qi3 jiang4 de shi2 jian1 .",
        "huan2 qiu2 wang3 bao4 dao4",
	"e2 luo2 si1 wei4 xing1 wang3 shi2 yi1 ri4 bao4 dao4 cheng1",
	"ji4 nian4 di4 yi2 ci4 shi4 jie4 da4 zhan4 jie2 shu4 yi4 bai3 zhou1 nian2 qing4 zhu4 dian3 li3 zai4 ba1 li2 ju3 xing2",
        "e2 luo2 si1 zong3 tong3 pu3 jing1 he2 mei3 guo2 zong3 tong3 te4 lang3 pu3 zai4 ba1 li2 kai3 xuan2 men2 jian4 mian4 shi2 wo4 shou3 zhi4 yi4",
        "pu3 jing1 biao3 shi4",
	"tong2 mei3 guo2 zong3 tong3 te4 lang3 pu3 jin4 xing2 le hen3 hao3 de jiao1 liu2",
        "e2 luo2 si1 zong3 tong3 zhu4 shou3 you2 li3",
	"wu1 sha1 ke1 fu1 biao3 shi4",
	"fa3 guo2 fang1 mian4 zhi2 yi4 yao1 qiu2 bu2 yao4 zai4 ba1 li2 ju3 xing2 ji4 nian4 huo2 dong4 qi1 jian1 ju3 xing2 e2 mei3 liang3 guo2 zong3 tong3 de dan1 du2 hui4 wu4",
        "da2 cheng2 le xie2 yi4",
	"wo3 men yi3 jing1 kai1 shi3 xie2 tiao2 e2 luo2 si1 he2 mei3 guo2 zong3 tong3 hui4 wu4 de shi2 jian1",
	"dan4 hou4 lai2 wo3 men kao3 lv4 dao4 le fa3 guo2 tong2 hang2 men de dan1 you1 he2 guan1 qie4",
	"wu1 sha1 ke1 fu1 shuo1",
        "yin1 ci3",
	"wo3 men yu3 mei3 guo2 dai4 biao3 men yi4 qi3 jin4 xing2 le tao3 lun4",
	"jue2 ding4 zai4 bu4 yi2 nuo4 si1 ai4 li4 si1 feng1 hui4 shang4 jin4 xing2 nei4 rong2 geng4 feng1 fu4 de dui4 hua4",
        "bao4 dao4 cheng1",
	"pu3 jing1 he2 te4 lang3 pu3 zai4 ai4 li4 she4 gong1 wu3 can1 hui4 shang4 de zuo4 wei4 an1 pai2 zai4 zui4 hou4 yi1 fen1 zhong1 jin4 xing2 le tiao2 zheng3",
	"dan4 zhe4 bing4 bu4 fang2 ai4 ta1 men jiao1 tan2",
        "sui1 ran2 dong1 dao4 zhu3 fa3 guo2 dui4 ta1 men zai4 ba1 li2 de hui4 wu4 biao3 shi4 fan3 dui4",
	"dan4 e2 mei3 ling3 dao3 ren2 reng2 ran2 biao3 shi4",
	"ta1 men xi1 wang4 zai4 ai4 li4 she4 gong1 de gong1 zuo4 wu3 can1 shang4 hui4 mian4",
        "chu1 bu4 zuo4 wei4 biao3 xian3 shi4",
	"te4 lang3 pu3 bei4 an1 pai2 zai4 pu3 jing1 pang2 bian1",
	"dan4 zai4 sui2 hou4 jin4 xing2 de gong1 zuo4 wu3 can1 qi1 jian1",
	"zuo4 wei4 an1 pai2 xian3 ran2 yi3 jing1 fa1 sheng1 le bian4 hua4",
        "cong2 zhao4 pian1 lai2 kan4",
	"pu3 jing1 dang1 shi2 zheng4 quan2 shen2 guan4 zhu4 de yu3 lian2 he2 guo2 mi4 shu1 chang2 gu3 te4 lei2 si1 jiao1 tan2",
	"ou1 meng2 wei3 yuan2 hui4 zhu3 xi2 rong2 ke4 zuo4 zai4 pu3 jing1 de you4 bian1",
        "er2 te4 lang3 pu3 ze2 zuo4 zai4 ma3 ke4 long2 pang2 bian1",
	"ma3 ke4 long2 de you4 bian1 ze2 shi4 de2 guo2 zong3 li3 mo4 ke4 er3",
        "ci3 qian2",
	"pu3 jing1 zai4 fang3 wen4 ba1 li2 qi1 jian1 biao3 shi4",
	"ta1 bu4 pai2 chu2 yu3 te4 lang3 pu3 zai4 gong1 zuo4 wu3 can1 shi2 jin4 xing2 jiao1 liu2",
        "pu3 jing1 zai4 fa3 guo2 pin2 dao4 de jie2 mu4 zhong1 hui2 da2 shi4 fou3 yi3 tong2 te4 lang3 pu3 jin4 xing2 jiao1 liu2 de wen4 ti2 shi2 biao3 shi4 zan4 shi2 mei2 you3",
	"wo3 men zhi3 da3 le ge4 zhao1 hu1",
        "yi2 shi4 yi3 zhe4 yang4 de fang1 shi4 jin4 xing2",
	"wo3 men wu2 fa3 zai4 na4 li3 jin4 xing2 jiao1 liu2",
	"wo3 men guan1 kan4 le fa1 sheng1 de shi4 qing2",
        "dan4 xian4 zai4 hui4 you3 gong1 zuo4 wu3 can1",
	"ye3 xu3 zai4 na4 li3 wo3 men hui4 jin4 xing2 jie1 chu4",
        "dan4 shi4",
	"wu2 lun4 ru2 he2",
	"wo3 men shang1 ding4",
	"wo3 men zai4 zhe4 li3 bu4 hui4 wei2 fan3 zhu3 ban4 guo2 de gong1 zuo4 an1 pai2",
	"gen1 ju4 ta1 men de yao1 qiu2",
        "wo3 men bu4 hui4 zai4 zhe4 li3 zu3 zhi1 ren4 he2 hui4 mian4",
	"er2 shi4 ke3 neng2 hui4 zai4 G er4 ling2 qi1 jian1 huo4 zai4 ci3 zhi1 hou4 ju3 xing2 hui4 mian4",
        "pu3 jing1 hai2 biao3 shi4",
	"e2 luo2 si1 zhun3 bei4 tong2 mei3 guo2 jin4 xing2 dui4 hua4",
	"fan3 zheng4 bu2 shi4 mo4 si1 ke1 yao4 tui4 chu1 zhong1 dao3 tiao2 yue1",

	# "guan1 yu2 xi1 zang4 de chuan2 shuo1 , you3 hen3 duo1 , li4 lai2 , dou1 shi4 chao2 sheng4 zhe3 de tian1 tang2 ,",
	# "er2 zuo4 wei2 , zhong1 guo2 xi1 nan2 bian1 chui2 zhong4 di4 , ye3 dou1 shi4 zhong1 guo2 ling3 tu3 , bu4 ke3 fen1 ge1 de yi1 bu4 fen4 .",
	# "er4 ling2 yi1 wu3 nian2 , yang1 shi4 ceng2 jing1 bo1 chu1 guo4 , yi1 bu4 gao1 fen1 ji4 lu4 pian4 , di4 san1 ji2 pian4 zhong1 , tian1 gao1 de kuo4 de feng1 jing3 ,",
	# "rang4 wu2 shu4 ren2 dui4 xi1 zang4 , qing2 gen1 shen1 zhong3 shi2 ge2 liang3 nian2 ,",
	# "you2 yuan2 ban1 ren2 ma3 da3 zao4 de zi3 mei4 pian1 , ji2 di4 qiao3 ran2 shang4 xian4 !",
	# "mei3 yi1 zheng4 dou1 shi4 bi4 zhi3 , mei3 yi2 mu4 dou1 shi4 ren2 jian1 xian1 jing4 .",
	# "zi4 ying3 pian1 bo1 chu1 zhi1 lai2 , hao3 ping2 ru2 chao2 , jiu4 lian2 yi2 xiang4 yi3 yan2 jin3 chu1 ming2 de dou4 ban4 ping2 fen1 , ye3 shi4 hen3 gao1 .",
	# "zao3 zai4 er4 ling2 yi1 wu3 nian2 , ta1 de di4 yi1 ji4 di4 san1 ji2 jiu4 na2 dao4 le dou4 ban4 jiu3 dian3 er4 fen1 .",
	# "er2 rang4 ta1 yi1 xia4 na2 dao4 jiu3 dian3 wu3 fen1 de yuan2 yin1 , shi4 yin1 wei4 , ta1 zhan3 shi4 le ,",
	# "zai4 na4 pian4 jue2 mei3 yu3 pin2 ji2 bing4 cun2 de jing4 tu3 shang4 , pu3 tong1 ren2 de zhen1 shi2 sheng1 huo2 , shi4 shen2 me yang4 zi .",
	]

	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
