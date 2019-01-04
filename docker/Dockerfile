FROM continuumio/anaconda3:latest
FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update
RUN apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools wget git vim

RUN wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
RUN tar -jxvf LJSpeech-1.1.tar.bz2

RUN git clone https://github.com/Rayhane-mamah/Tacotron-2.git

WORKDIR Tacotron-2
RUN ln -s ../LJSpeech-1.1 .
RUN pip install -r requirements.txt