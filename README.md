# Tacotron-2:
Tensorflow implementation of Deep mind's Tacotron-2. A deep neural network architecture described in this paper: [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)


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
	│   ├── mel-spectrograms
	│   ├── plots
	│   ├── pretrained
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
	├── training_data	(1)
	│   ├── audio
	│   └── mels
	└── wavenet_vocoder
		└── models




The previous tree shows what the current state of the repository.

- Step **(0)**: Get your dataset, here I have set the examples of **Ljspeech**, **en_US** and **en_UK** (from **M-AILABS**).
- Step **(1)**: Preprocess your data. This will give you the **training_data** folder.
- Step **(2)**: Train your Tacotron model. Yields the **logs-Tacotron** folder.
- Step **(3)**: Synthesize/Evaluate the Tacotron model. Gives the **tacotron_output** folder.


Note:
- **Our preprocessing only supports Ljspeech and Ljspeech-like datasets (M-AILABS speech data)!** If running on datasets stored differently, you will probably need to make your own preprocessing script.
- In the previous tree, files **were not represented** and **max depth was set to 3** for simplicity.

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
first, you need to have python 3 installed along with [Tensorflow v1.6](https://www.tensorflow.org/install/).

next you can install the requirements. If you are an Anaconda user:

> pip install -r requirements.txt

else:

> pip3 install -r requirements.txt

# Dataset:
We tested the code above on the [ljspeech dataset](https://keithito.com/LJ-Speech-Dataset/), which has almost 24 hours of labeled single actress voice recording. (further info on the dataset are available in the README file when you download it)

We are also running current tests on the [new M-AILABS speech dataset](http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/) which contains more than 700h of speech (more than 80 Gb of data) for more than 10 languages.

After **downloading** the dataset, **extract** the compressed file, and **place the folder inside the cloned repository.**

# Preprocessing
Before running the following steps, please make sure you are inside **Tacotron-2 folder**

> cd Tacotron-2

Preprocessing can then be started using: 

> python preprocess.py

or 

> python3 preprocess.py

dataset can be chosen using the **--dataset** argument. If using M-AILABS dataset, you need to provide the **language, voice, reader, merge_books and book arguments** for your custom need. Default is **Ljspeech**.

Example M-AILABS:

> python preprocess.py --dataset='M-AILABS' --language='en_US' --voice='female' --reader='mary_ann' --merge_books=False --book='northandsouth'

or if you want to use all books for a single speaker:

> python preprocess.py --dataset='M-AILABS' --language='en_US' --voice='female' --reader='mary_ann' --merge_books=True

This should take no longer than a **few minutes.**

# Training:
Feature prediction model can be **trained** using:

> python train.py --model='Tacotron'

or 

> python3 train.py --model='Tacotron'

checkpoints will be made each **100 steps** and stored under **logs-Tacotron folder.**

Naturally, **training the wavenet** is done by: (Not implemented yet)

> python train.py --model='Wavenet'

or 

> python3 train.py --model='Wavenet'

logs will be stored inside **logs-Wavenet**.

**Note:**
- If model argument is not provided, training will default to Tacotron model training.

# Synthesis
There are **three types** of mel spectrograms synthesis for the Spectrogram prediction network (Tacotron):

- **Evaluation** (synthesis on custom sentences). This is what we'll usually use after having a full end to end model.

> python synthesize.py --model='Tacotron' --mode='eval'

or

> python3 synthesize.py --model='Tacotron' --mode='eval'

- **Natural synthesis** (let the model make predictions alone by feeding last decoder output to the next time step).

> python synthesize.py --model='Tacotron' --GTA=False

or

> python3 synthesize.py --model='Tacotron' --GTA=False

- **Ground Truth Aligned synthesis** (DEFAULT: the model is assisted by true labels in a teacher forcing manner). This synthesis method is used when predicting mel spectrograms used to train the wavenet vocoder. (yields better results as stated in the paper)

> python synthesize.py --model='Tacotron'

or 

> python3 synthesize.py --model='Tacotron'

Synthesizing the waveforms conditionned on previously synthesized Mel-spectrograms can be done with:

> python synthesize.py --model='Wavenet'

or 

> python3 synthesize.py --model='Wavenet'

**Note:**
- If model argument is not provided, synthesis will default to Tacotron model synthesis.
- If mode argument is not provided, synthesis defaults to Ground Truth Aligned synthesis.

# Pretrained model and Samples:
Pre-trained models and audio samples will be added at a later date due to technical difficulties. You can however check some primary insights of the model performance (at early stages of training) [here](https://github.com/Rayhane-mamah/Tacotron-2/issues/4#issuecomment-378741465).


# References and Resources:
- [Tensorflow original tacotron implementation](https://github.com/keithito/tacotron)
- [Original tacotron paper](https://arxiv.org/pdf/1703.10135.pdf)
- [Attention-Based Models for Speech Recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)

**Work in progress**
