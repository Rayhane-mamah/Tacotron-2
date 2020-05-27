FROM continuumio/anaconda3:latest
FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update
RUN apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg wget git vim

RUN wget http://launchpadlibrarian.net/339874908/libav-tools_3.3.4-2_all.deb
RUN dpkg -i libav-tools_3.3.4-2_all.deb

RUN wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
RUN tar -jxvf LJSpeech-1.1.tar.bz2

RUN git clone https://github.com/Cyancey96/Tacotron-2.git

WORKDIR Tacotron-2
RUN ln -s ../LJSpeech-1.1 .
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==1.10
RUN pip3 install -r ./requirements.txt
