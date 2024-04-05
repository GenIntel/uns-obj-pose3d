# ubuntu20.04, python3.8, cuda11.7
python3 -m venv venv_od3d
source venv_od3d/bin/activate
pip3 install pip --upgrade
CUDA_HOME=/misc/software/cuda/cuda-11.7
PATH=${CUDA_HOME}/bin:${PATH}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
export FORCE_CUDA=1
export CUDA_HOME
export LD_LIBRARY_PATH
export FORCE_CUDA
export TORCH_CUDA_ARCH_LIST
export PATH
#./misc/software/cuda/add_environment_cuda11.7.sh

pip install -U wheel
pip install -U fvcore
pip install -U iopath
pip install Cython
export PATH
export LD_LIBRARY_PATH
export CUDA_HOME

# default torch uses CUDA 12.x which is then not compatible with installing pytorch3d
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install -e .

# add configs:
#  - credentials/default.yaml
#  - platform/local.yaml
#  - platform/slurm.yaml
