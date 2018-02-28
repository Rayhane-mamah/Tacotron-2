# Tacotron-2
Tensorflow implementation of Deep mind's Tacotron-2. A deep neural network architecture described in this paper: [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)


# Current state:

To have an overview of our advance on this project, please refer to [this discussion](https://github.com/Rayhane-mamah/Tacotron-2/issues/4)

since the two parts of the global model are trained separately, we can start by training the feature prediction model to use his predictions later during the wavenet training.

# How to start
first, you need to have python 3 installed along with [Tensorflow](https://www.tensorflow.org/install/).

next you can install the requirements using:

> pip install -r requirements.txt

# Dataset:
We tested the code above on the [ljspeech dataset](https://keithito.com/LJ-Speech-Dataset/), which has almost 24 hours of labeled single actress voice recording. (further info on the dataset are available in the README file when you download it)

After downloading the dataset, extract the compressed file, and place the folder in the same folder of the cloned repository.
after that launch the preprocessing file:

> python preprocess.py

This should take few minutes.

# Training:
You can start the training using:

> python train.py

checkpoints will be made each 100 steps and stored under logs-<model_name> folder.

# Synthesis
There are three types of mel spectrograms synthesis using this model:

- Evaluation (synthesis on custom sentences). This is what we'll usually use after having a full end to end model.

> python synthesize.py --mode='eval'

- Natural synthesis (let the model make predictions alone by feeding last decoder output to the next time step).

> python synthesize.py --GTA=False

- Ground Truth Aligned synthesis (DEFAULT: the model is assisted by true labels in a teacher forcing manner). This synthesis method is used when predicting mel spectrograms used to train the wavenet vocoder. (yields better results as states in the paper)

> python synthesize.py

## DISCLAIMER:
Due to some constraints, we won't be able to provide a pretrained feature prediction model at the moment. Vocoder Wavenet model however is in development state. In the mean time, if someone can train a feature prediction model, we will gladly add it to the repository.

## Note:
Due to the large size of audio samples in the dataset, we advise you to drop the batch size to 32 or lower (depending on your gpu load). Please keep in mind that this will slow the training process.

# References:
- [Tensorflow original tacotron implementation](https://github.com/keithito/tacotron)
- [Original tacotron paper](https://arxiv.org/pdf/1703.10135.pdf)
- [Attention-Based Models for Speech Recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)


** Work in progress, further info will be added **

** This work is independant from deep mind **
