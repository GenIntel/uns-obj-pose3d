###### SETUP DOCKER WITHOUT ROOT AND GPU ######

#sudo usermod -aG docker ${USER}

#curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
#curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
#    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
#sudo apt-get update
#sudo apt-get install -y nvidia-container-toolkit
#sudo systemctl restart docker


###### BUILD ######

# docker build . -t limpbot/droid-slam:v1 --build-arg UID=$(id -u) --build-arg GID=$(id -g)
#docker build . -t limpbot/droid-slam:v1 -f ../envs/DROID-SLAM/Dockerfile
#
####### RUN ######
#
#SEQ=206_21810_45890
#PATH_IN=/misc/lmbraid19/sommerl/datasets/CO3D/car/${SEQ}/images
#PATH_OUT=/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/droid_slam
#RPATH_OUT=car/${SEQ}
#STRIDE=1
#
## droid slam
#docker run --user=$(id -u):$(id -g) --gpus all -e RPATH_OUT=${RPATH_OUT} -e STRIDE=${STRIDE} -v ${PATH_IN}:/home/appuser/in -v ${PATH_OUT}:/home/appuser/DROID-SLAM/reconstructions/out -t limpbot/droid-slam:v1

# interactive mode
# docker run --user=$(id -u):$(id -g) --gpus all -e RPATH_OUT=${RPATH_OUT} -e STRIDE=${STRIDE} -v ${PATH_IN}:/home/appuser/in -v ${PATH_OUT}:/home/appuser/DROID-SLAM/reconstructions/out -it limpbot/droid-slam:v1


FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#ARG UID=1000
#ARG GID=1000

# Metadata
#LABEL maintainer="rajahaseeb147@gmail.com" \
#    project-name="Droid-SLAM" \
#    com.droid-slam.version="0.0.1-alpha" \
#    com.droid-slam.realease-data="2023.01.20"

# Set arguments and env variables
ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DIVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Fetch nvidia signing keys
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

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


#RUN adduser --system --group appuser
ENV UID=19104
ENV GID=1061
RUN groupadd -g "${GID}" appuser && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" appuser

# --group # --no-create-home appuser
USER appuser

# Set working directory
#RUN mkdir -p /root/
WORKDIR /home/appuser/

# Set user to root to avoid permission issues
USER appuser

# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/home/appuser/miniconda3/bin:${PATH}"

# Install Droid SLAM dependencies
RUN conda install python=3.9

RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y

RUN pip install gdown matplotlib open3d opencv-python torch-scatter tensorboard scipy tqdm pyyaml

RUN conda install -c conda-forge suitesparse -y

# Clone repository
WORKDIR /home/appuser/
#RUN echo "BLUB"
#RUN git clone --recursive https://github.com/pytholic/DROID-SLAM.git
#RUN git clone --recursive https://github.com/limpbot/DROID-SLAM.git

WORKDIR /home/appuser/DROID-SLAM
USER root
RUN chown -R appuser:appuser /home/appuser/DROID-SLAM
USER appuser
RUN gdown https://drive.google.com/uc?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh

ADD . /home/appuser/DROID-SLAM

USER root
RUN chown -R appuser:appuser /home/appuser/DROID-SLAM
USER appuser

# Install extensions
RUN python setup.py install


ENV STRIDE="1"
ENV RPATH_OUT="seq"
ENV PATH_IN="/home/appuser/in"
ENV PATH_OUT="/home/appuser/DROID-SLAM/reconstructions/out"

RUN mkdir -p $PATH_IN
RUN mkdir -p $PATH_OUT

USER root
#RUN groupmod -g "${GID}" appuser && usermod -u "${UID}" -g "${GID}" appuser
RUN groupmod -g "1061" appuser && usermod -u "19104" -g "1061" appuser
USER appuser

# use this for interactive container
#ENTRYPOINT [ "/bin/bash" ]

CMD ["sh", "-c", "python demo.py --imagedir=${PATH_IN} --calib=${PATH_OUT}/${RPATH_OUT}/calib.txt --reconstruction_path out/${RPATH_OUT} --weights droid.pth --stride ${STRIDE} --disable_vis"]
