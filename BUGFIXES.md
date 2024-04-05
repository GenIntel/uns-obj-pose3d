


# BUG 1
### Symptom
    cv2.imshow(...) freeze
### Cause
    python packages av and opencv-python clash (imshow freezes if av is installed)
### Solution
    pip uninstall av


# BUG 2
### Symptom
    RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the ‘spawn’ start method   
### Cause
    Sharing GPU memory across multiprocesses   
### Solution
    Don't use GPU with dataloader


# BUG 3

### Symptom (installing pytorch3d)
    build_ext
        error: [Errno 2] No such file or directory: '/usr/local/cuda/bin/nvcc'

### Cause
    nvcc not available in nvidia runtime image
### Solution 
    use nvidia devel image, e.g. nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04


# BUG 4

[Open3D WARNING] GLFW Error: X11: The DISPLAY environment variable is missing
[Open3D WARNING] Failed to initialize GLFW
Segmentation fault (core dumped)

export DISPLAY=:0.0; 

# BUG 5

end of file reading mesh

/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/mesh/alpha500/remote/107_12754_22765/mesh.ply


/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/mesh/alpha500/remote/107_12754_22765/mesh.ply
107_12754_22765
105_12562_23650

# CUDNN=cudnn-linux-x86_64-8.9.5.29_cuda11-archive
#cp ${CUDNN}/include/cudnn*.h ${CUDA_HOME}/include
#cp ${CUDNN}/lib/libcudnn* ${CUDA_HOME}/lib64
#chmod a+r ${CUDA_HOME}/include/cudnn*.h ${CUDA_HOME}/lib64/libcudnn*

# BUG 6

wrong cuda device

CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER
if CUDA_DEVICE_ORDER is not set, 