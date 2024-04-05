# docker build -f docker/od3d/Dockerfile -t limpbot/od3d:v1 .
# docker run -it --gpus all -u $(id -u):$(id -g) limpbot/od3d:v1

# note: devel is required to have nvcc
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

ENV NVIDIA_VISIBLE_DIVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Fetch nvidia signing keys
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# Get the basic Linux packages upgraded and get the development tools setup
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y wget \
    build-essential \
    manpages-dev \
    ca-certificates \
    g++ \
    git \
    libglfw3-dev \
    libgles2-mesa-dev \
    && apt-get clean  \
    && gcc --version

# OpenCV requires
RUN apt-get install libglib2.0-0

RUN apt-get update

RUN apt-get install -y python3.10 python3.10-venv python3-pip
RUN pip install pip --upgrade

# option 1: copy from local
ADD . /od3d/
# option 2: clone from git
# RUN git clone https://github.com/Generative-Vision-Robust-Learning/od3d.git

WORKDIR od3d

RUN python3 -m venv venv_od3d
RUN . venv_od3d/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# note: not sure when this is required
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
# note: this is required at least for pycuda and pytorch3d
ENV CUDA_HOME=/usr/local/cuda

RUN . venv_od3d/bin/activate && pip install -e .

# use this for interactive container
ENTRYPOINT ["/bin/bash", "-c", "source venv/bin/activate"]
