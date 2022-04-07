#10.2-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda@sha256:0d9e949e0d158e9d5ec0e402c222e0b9bcf8f9b6d58fc43fb91617bdefd796f4

RUN apt-get update -q \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y \
	curl \
	git \
	unzip \
	libgl1-mesa-dev \
	libgl1-mesa-glx \
	libglew-dev \
	libosmesa6-dev \
	software-properties-common \
	net-tools \
	vim \
	wget \
	xpra \
	xvfb \
	patchelf \
	ffmpeg \
	cmake \
	swig \
	xserver-xorg-dev \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*



RUN mkdir /root/.mujoco && \
    cd /root/.mujoco  && \
    curl -O https://www.roboti.us/download/mujoco200_linux.zip && \
    unzip mujoco200_linux.zip && \
    echo DUMMY_KEY > /root/.mujoco/mjkey.txt

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro200/bin