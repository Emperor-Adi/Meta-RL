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

#Mujoco Install
RUN mkdir /root/.mujoco && \
    cd /root/.mujoco  && \
    curl -O https://www.roboti.us/download/mujoco200_linux.zip && \
    unzip mujoco200_linux.zip && \
    echo DUMMY_KEY > /root/.mujoco/mjkey.txt && \
	rm mujoco200_linux.zip

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200_linux/bin
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

#Get Repo
RUN cd /root/ && \
	git clone https://github.com/TheMonocledHamster/Meta-RL.git

#Setup Conda Environment
RUN cd /root/ && \
	wget "https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh" && \
	bash Miniconda3-py38_4.11.0-Linux-x86_64.sh
# RUN rm Miniconda3-py38_4.11.0-Linux-x86_64.sh
# RUN conda init
# RUN conda config --set auto_activate_base false
# RUN conda config --append channels conda-forge
# RUN conda create --name RL python=3.6 numpy=1.19 tensorflow-gpu=2
# RUN conda activate RL
