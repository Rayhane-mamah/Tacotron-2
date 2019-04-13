# Tacotron-2:
Tensorflow implementation of DeepMind's Tacotron-2. A deep neural network architecture described in this paper: [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)

This Repository contains additional improvements and attempts over the paper, we thus propose **paper_hparams.py** file which holds the exact hyperparameters to reproduce the paper results without any additional extras.

Suggested **hparams.py** file which is default in use, contains the hyperparameters with extras that proved to provide better results in most cases. Feel free to toy with the parameters as needed.

DIFFERENCES WILL BE HIGHLIGHTED IN DOCUMENTATION SHORTLY.


# Repository Structure:
	Tacotron-2
	├── datasets
	├── en_UK		(0)
	│   └── by_book
	│       └── female
	├── en_US		(0)
	│   └── by_book
	│       ├── female
	│       └── male
	├── LJSpeech-1.1	(0)
	│   └── wavs
	├── logs-Tacotron	(2)
	│   ├── eval_-dir
	│   │ 	├── plots
	│ 	│ 	└── wavs
	│   ├── mel-spectrograms
	│   ├── plots
	│   ├── taco_pretrained
	│   ├── metas
	│   └── wavs
	├── logs-Wavenet	(4)
	│   ├── eval-dir
	│   │ 	├── plots
	│ 	│ 	└── wavs
	│   ├── plots
	│   ├── wave_pretrained
	│   ├── metas
	│   └── wavs
	├── logs-Tacotron-2	( * )
	│   ├── eval-dir
	│   │ 	├── plots
	│ 	│ 	└── wavs
	│   ├── plots
	│   ├── taco_pretrained
	│   ├── wave_pretrained
	│   ├── metas
	│   └── wavs
	├── papers
	├── tacotron
	│   ├── models
	│   └── utils
	├── tacotron_output	(3)
	│   ├── eval
	│   ├── gta
	│   ├── logs-eval
	│   │   ├── plots
	│   │   └── wavs
	│   └── natural
	├── wavenet_output	(5)
	│   ├── plots
	│   └── wavs
	├── training_data	(1)
	│   ├── audio
	│   ├── linear
	│	└── mels
	└── wavenet_vocoder
		└── models


The previous tree shows the current state of the repository (separate training, one step at a time).

- Step **(0)**: Get your dataset, here I have set the examples of **Ljspeech**, **en_US** and **en_UK** (from **M-AILABS**).
- Step **(1)**: Preprocess your data. This will give you the **training_data** folder.
- Step **(2)**: Train your Tacotron model. Yields the **logs-Tacotron** folder.
- Step **(3)**: Synthesize/Evaluate the Tacotron model. Gives the **tacotron_output** folder.
- Step **(4)**: Train your Wavenet model. Yield the **logs-Wavenet** folder.
- Step **(5)**: Synthesize audio using the Wavenet model. Gives the **wavenet_output** folder.

- Note: Steps 2, 3, and 4 can be made with a simple run for both Tacotron and WaveNet (Tacotron-2, step ( * )).


Note:
- **Our preprocessing only supports Ljspeech and Ljspeech-like datasets (M-AILABS speech data)!** If running on datasets stored differently, you will probably need to make your own preprocessing script.
- In the previous tree, files **were not represented** and **max depth was set to 3** for simplicity.
- If you run training of both **models at the same time**, repository structure will be different.

# Pretrained model and Samples:
Pre-trained models and audio samples will be added at a later date. You can however check some primary insights of the model performance (at early stages of training) [here](https://github.com/Rayhane-mamah/Tacotron-2/issues/4#issuecomment-378741465). THIS IS VERY OUTDATED, I WILL UPDATE THIS SOON

# Model Architecture:
<p align="center">
  <img src="https://preview.ibb.co/bU8sLS/Tacotron_2_Architecture.png"/>
</p>

The model described by the authors can be divided in two parts:
- Spectrogram prediction network
- Wavenet vocoder

To have an in-depth exploration of the model architecture, training procedure and preprocessing logic, refer to [our wiki](https://github.com/Rayhane-mamah/Tacotron-2/wiki)

# Current state:

To have an overview of our advance on this project, please refer to [this discussion](https://github.com/Rayhane-mamah/Tacotron-2/issues/4)

since the two parts of the global model are trained separately, we can start by training the feature prediction model to use his predictions later during the wavenet training.

# How to start
- **Machine Setup:**

First, you need to have python 3 installed along with [Tensorflow](https://www.tensorflow.org/install/).

Next, you need to install some Linux dependencies to ensure audio libraries work properly:

> apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools

Finally, you can install the requirements. If you are an Anaconda user: (else replace **pip** with **pip3** and **python** with **python3**)

> pip install -r requirements.txt

- **Docker:**

Alternatively, one can build the **docker image** to ensure everything is setup automatically and use the project inside the docker containers.
**Dockerfile is insider "docker" folder**

docker image can be built with:

> docker build -t tacotron-2_image docker/

Then containers are runnable with:

> docker run -i --name new_container tacotron-2_image

Please report any issues with the Docker usage with our models, I'll get to it. Thanks!

# Dataset:
We tested the code above on the [ljspeech dataset](https://keithito.com/LJ-Speech-Dataset/), which has almost 24 hours of labeled single actress voice recording. (further info on the dataset are available in the README file when you download it)

We are also running current tests on the [new M-AILABS speech dataset](http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/) which contains more than 700h of speech (more than 80 Gb of data) for more than 10 languages.

After **downloading** the dataset, **extract** the compressed file, and **place the folder inside the cloned repository.**

# Hparams setting:
Before proceeding, you must pick the hyperparameters that suit best your needs. While it is possible to change the hyper parameters from command line during preprocessing/training, I still recommend making the changes once and for all on the **hparams.py** file directly.

To pick optimal fft parameters, I have made a **griffin_lim_synthesis_tool** notebook that you can use to invert real extracted mel/linear spectrograms and choose how good your preprocessing is. All other options are well explained in the **hparams.py** and have meaningful names so that you can try multiple things with them.

AWAIT DOCUMENTATION ON HPARAMS SHORTLY!!

# Preprocessing
Before running the following steps, please make sure you are inside **Tacotron-2 folder**

> cd Tacotron-2

Preprocessing can then be started using: 

> python preprocess.py

dataset can be chosen using the **--dataset** argument. If using M-AILABS dataset, you need to provide the **language, voice, reader, merge_books and book arguments** for your custom need. Default is **Ljspeech**.

Example M-AILABS:

> python preprocess.py --dataset='M-AILABS' --language='en_US' --voice='female' --reader='mary_ann' --merge_books=False --book='northandsouth'

or if you want to use all books for a single speaker:

> python preprocess.py --dataset='M-AILABS' --language='en_US' --voice='female' --reader='mary_ann' --merge_books=True

This should take no longer than a **few minutes.**

# Training:
To **train both models** sequentially (one after the other):

> python train.py --model='Tacotron-2'


Feature prediction model can **separately** be **trained** using:

> python train.py --model='Tacotron'

checkpoints will be made each **5000 steps** and stored under **logs-Tacotron folder.**

Naturally, **training the wavenet separately** is done by:

> python train.py --model='WaveNet'

logs will be stored inside **logs-Wavenet**.

**Note:**
- If model argument is not provided, training will default to Tacotron-2 model training. (both models)
- Please refer to train arguments under [train.py](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/train.py) for a set of options you can use.
- It is now possible to make wavenet preprocessing alone using **wavenet_proprocess.py**.

# Synthesis
To **synthesize audio** in an **End-to-End** (text to audio) manner (both models at work):

> python synthesize.py --model='Tacotron-2'

For the spectrogram prediction network (separately), there are **three types** of mel spectrograms synthesis:

- **Evaluation** (synthesis on custom sentences). This is what we'll usually use after having a full end to end model.

> python synthesize.py --model='Tacotron'

- **Natural synthesis** (let the model make predictions alone by feeding last decoder output to the next time step).

> python synthesize.py --model='Tacotron' --mode='synthesis' --GTA=False


- **Ground Truth Aligned synthesis** (DEFAULT: the model is assisted by true labels in a teacher forcing manner). This synthesis method is used when predicting mel spectrograms used to train the wavenet vocoder. (yields better results as stated in the paper)

> python synthesize.py --model='Tacotron' --mode='synthesis' --GTA=True

Synthesizing the **waveforms** conditionned on previously synthesized Mel-spectrograms (separately) can be done with:

> python synthesize.py --model='WaveNet'

**Note:**
- If model argument is not provided, synthesis will default to Tacotron-2 model synthesis. (End-to-End TTS)
- Please refer to synthesis arguments under [synthesize.py](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/synthesize.py) for a set of options you can use.


# References and Resources:
- [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)
- [Original tacotron paper](https://arxiv.org/pdf/1703.10135.pdf)
- [Attention-Based Models for Speech Recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Wavenet: A generative model for raw audio](https://arxiv.org/pdf/1609.03499.pdf)
- [Fast Wavenet](https://arxiv.org/pdf/1611.09482.pdf)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- [keithito/tacotron](https://github.com/keithito/tacotron)

