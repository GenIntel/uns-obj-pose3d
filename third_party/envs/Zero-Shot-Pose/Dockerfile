#  docker build -f third_party/envs/Zero-Shot-Pose/Dockerfile -t limpbot/zsp:v1 third_party/zero-shot-pose

#--build-arg ssh_prv_key="$(cat ~/.ssh/id_rsa)" --build-arg ssh_pub_key="$(cat ~/.ssh/id_rsa.pub)" --squash
#--build-arg UID=$(id -u) --build-arg GID=$(id -g)

# docker run  --gpus all -p 5000:5000 -it limpbot/zsp:v1
# docker run  --gpus all -p 5000:5000 -t limpbot/zsp:v1
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set arguments and env variables
ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DIVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Fetch nvidia signing keys
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

RUN apt-get update -y
RUN apt-get upgrade -y
# Install ubuntu base packages
RUN apt-get update && apt install -y --no-install-recommends \
    software-properties-common \
    apt-utils \
    dbus-x11 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    libgl1-mesa-dev \
    libglew-dev


RUN apt-get update && apt install -y --no-install-recommends \
    build-essential \
    libboost-all-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ninja-build

# Installing required utilities
RUN apt-get update && apt-get install -y \
    curl \
    git \
    ssh \
    unzip \
    vim \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get install wget -y
RUN apt-get install git -y

WORKDIR /root

# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

## prioritize 'conda-forge' channel
#RUN conda install -f environment.yml
RUN conda config --add channels conda-forge

## update existing packages to use 'conda-forge' channel
RUN conda update -n base --all

## install 'mamba'
RUN conda install -n base mamba
#RUN mamba init
## Make RUN commands use `bash --login`:
#SHELL ["/bin/bash", "--login", "-c"]
#RUN conda init bash
# Install Droid SLAM dependencies
#RUN mamba install python=3.9

#RUN mamba install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y

WORKDIR /root/zsp/
ADD environment.yml /root/zsp

RUN wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth

RUN mamba env create -f environment.yml

ENV PATH /root/miniconda3/envs/zsp/bin:$PATH
RUN echo "source activate zsp" > ~/.bashrc

RUN mamba update -y pytorch3d -c pytorch3d
# RUN conda activate zsp


RUN pip install 'pillow<7'

# RUN mamba install pytorch==1.9.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y

# RUN conda activate zsp
#RUN mamba install -y pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
#RUN mamba install -y -c fvcore -c iopath -c conda-forge fvcore iopath
#RUN mamba install -y -c bottler nvidiacub

# RUN pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1
#RUN pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
#RUN pip install torch torchvision torchaudio
#RUN RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y

# RUN apt install libpng libjpeg
# RUN apt install libpng12-0
# -dev

RUN pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

#RUN apt install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools
#RUN pip install wheel
#RUN export LD_LIBRARY="$LD_LIBRARY_PATH:/root/miniconda3/lib"
RUN pip install Flask # uwsgi #  flask

#RUN apt-get -y install python3-urllib3
#RUN pip install urllib3 --upgrade

ADD . /root/zsp
RUN pip install -e .

ENTRYPOINT ["env", "FLASK_APP=flask_server.py", "python", "-m", "flask", "run", "--host=0.0.0.0"]
#ENTRYPOINT ["env", "FLASK_APP=flask_server_test.py", "python", "-m", "flask", "run", "--host=0.0.0.0"]


# use this for interactive container
# ENTRYPOINT [ "/bin/bash" ]

# env FLASK_APP=flask_server_test.py python -m flask run --host=0.0.0.0

# ENTRYPOINT ["FLASK_ENV=development", "FLASK_APP=flask_server.py", "flask", "run"]
# FLASK_ENV=development FLASK_APP=flask_server_test.py flask run
# FLASK_ENV=development FLASK_APP=flask_server.py flask run

# env FLASK_APP=flask_server.py python -m flask run