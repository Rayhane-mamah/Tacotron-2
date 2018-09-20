FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /root

# install pyaudio library
RUN apt-get update \
    && apt-get install -y python3-pyaudio \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install other requirements
COPY requirements.txt requirements.txt
RUN grep -v '^pyaudio' requirements.txt > requirements_updated.txt \
    && pip3 install -r requirements_updated.txt
