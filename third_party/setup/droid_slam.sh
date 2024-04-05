OD3D_DIR=/scratch/sommerl/repos/NeMo
ENV_CONFIG_PATH=${OD3D_DIR}/third_party/envs/droidenv2.yaml
ENV_CONFIG_PATH=${OD3D_DIR}/third_party/DROID-SLAM/environment.yaml

DROID_SLAM_DIR=${OD3D_DIR}/third_party/DROID-SLAM

git submodule update --init --recursive

# note: critical to also have installed (copied) cudnn files at this cuda location
CUDA_HOME=/scratch/sommerl/cudas/cuda-11.8
export CUDA_HOME
PATH=${PATH}:${CUDA_HOME}/bin
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64 # ${LD_LIBRARY_PATH}:
export PATH
export LD_LIBRARY_PATH

# setting CUDA_HOME from conda
mamba env config vars set CUDA_HOME=$CUDA_HOME

cd ${DROID_SLAM_DIR}
deactivate
mamba env create -f ${ENV_CONFIG_PATH} --force
mamba activate droidenv
mamba update ffmpeg

pip install evo --upgrade --no-binary evo
pip install gdown


cd third_party/DROID-SLAM && python setup.py install && cd ../..

# try without
pip install open3d
# install od3d
pip install -e .

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
nvcc --version
