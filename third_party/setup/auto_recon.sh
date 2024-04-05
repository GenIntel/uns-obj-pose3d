# cd third_party && git submodule add git@github.com:zju3dv/AutoRecon.git third_party

git submodule update --init --recursive

mamba update -n base -c defaults conda # (4.x versus 23.x ???)
conda install -c conda-forge mamba

deactivate
mamba create --name auto_recon -y python=3.9
mamba activate auto_recon
python -m pip install --upgrade pip setuptools

CUDA_HOME=/scratch/sommerl/cudas/cuda-11.7 # environment variable is important, s.t. pytorch3d gets installed with gpu support, and tiny-cuda-nn
export CUDA_HOME
#TCNN_CUDA_ARCHITECTURES=86
#export TCNN_CUDA_ARCHITECTURES
PATH=${PATH}:${CUDA_HOME}/bin
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
export PATH
export LD_LIBRARY_PATH

# setting CUDA_HOME from conda
mamba env config vars set CUDA_HOME=$CUDA_HOME

# cuda 11.8 is required for tiny-cuda-nn, this contradicts that the repo was tested with cuda11.7
mamba install cudatoolkit=11.7 -c nvidia
mamba config --add channels conda-forge
mamba config --set channel_priority strict
mamba install colmap
mamba install -c conda-forge faiss-gpu

# bad practice but required
pip install faiss-gpu
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 torchaudio==0.13.1+cu118
pip install fvcore

pip install pytorch3d@git+https://github.com/facebookresearch/pytorch3d
pip install nvdiffrast@git+https://github.com/NVlabs/nvdiffrast

cd third_party/AutoRecon/third_party/AutoDecomp/third_party/Hierarchical-Localization && pip install -e . && cd ../../../../../..
cd third_party/AutoRecon/third_party/AutoDecomp/third_party/LoFTR && pip install -e . && cd ../../../../../..
cd third_party/AutoRecon/third_party/AutoDecomp && pip install -e . && cd ../../../..

# weights for LoFTR
gdown https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp -O third_party/AutoRecon/third_party/AutoDecomp/third_party/LoFTR/weights --folder

export TCNN_CUDA_ARCHITECTURES=61 # gpu has 86 and 61 compatibility, but 86 is not available as software (maybe due to cuda11.7 or pytorch1.13.1)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch  --force-reinstall --no-cache-dir

cd third_party/AutoRecon && pip install -e . && cd ../.. && cd ../..

# install tab completion
ns-install-cli

pip install torch_cluster --force-reinstall --no-cache-dir
#pip install torch_cluster --force-reinstall --no-cache-dir

#pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
#--extra-index-url https://download.pytorch.org/whl/cu117
# --extra-index-url https://download.pytorch.org/whl/cu118


#pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# deprecated usage of torch._sixx
#pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --upgrade --force-reinstall
# fixed by removing torch._sixx
#pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
# maybe have to go to: CUDA 11.8, nvidia-driver 520, pytorch 1.13.1, + export TCNN_CUDA_ARCHITECTURES=86

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
nvcc --version
