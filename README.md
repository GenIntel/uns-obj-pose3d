
## [Unsupervised Learning of Category-Level 3D Pose from Object-Centric Videos [CVPR'24]](https://generative-vision-robust-learning.github.io/uns-obj-pose3d)

```commandline
@InProceedings{ Sommer_2024_CVPR, 
 author    = {Sommer, Leonhard and Jesslen, Artur and Ilg, Eddy and Kortylewski, Adam}, 
 title     = {Unsupervised Learning of Category-Level 3D Pose from Object-Centric Videos}, 
 booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
 month     = {June}, 
 year      = {2024}, 
 pages     = {22787-22796} 
 }
```

### Install

#### Environment
```
python3 -m venv venv_od3d
source venv_od3d/bin/activate
pip3 install pip --upgrade
CUDA_HOME=/misc/software/cuda/cuda-11.7
export CUDA_HOME

pip install wheel # pytorch3d requires this
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### Install in Editable Mode
```
git submodule update --init --recursive
pip install -e .
```

#### Install in Non-editable mode

```
pip install git@github.com:Generative-Vision-Robust-Learning/od3d.git  
```

### Configure 

- Platform
  - `config/platform/local.yaml`  
  - `config/platform/torque.yaml`  
  - `config/platform/slurm.yaml`  
  - `~/.ssh/config` with `config/platform/ssh-config-template`  
    - Verify
      - `od3d platform run -p [torque|slurm]`
- Wandb
  - `wandb init`
- Credentials (optional)
    - `config/credentials/default.yaml`

    
### Dataset Setup

  1) Download
     - `od3d dataset setup -d [co3d|pascal3d|objectnet3d]`
  2) Extract Meta Data per Frame/Sequence (e.g. camera, quality, etc.)
     - `od3d dataset extract-meta -d [co3d|pascal3d|objectnet3d]`
  3) Preprocess (e.g. point cloud, mesh, masks, etc.)
     - `od3d dataset preprocess -d [co3d|pascal3d|objectnet3d]`

  - Visualize
    - `od3d dataset visualize -d [co3d|pascal3d|objectnet3d]`

  - Synchronize the target with the source platform
    - `od3d dataset rsync -s local -t slurm -d co3d`  
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s local -t torque -d co3d"`  
    - `export MESH_TYPE=alpha500`
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s local -t torque -d co3d -r mesh/$(MESH_TYPE)"`        
    - `export MESH_FEATURE_TYPE=M_dinov2_vits14_frozen_base_no_norm_T_centerzoom512_R_acc`
    - `export ALIGNED_TYPE=aligned_N_alpha500_dinov2s`
    - `export MESH_FEATURE_TYPE=M_dinov2_vitb14_frozen_base_no_norm_T_centerzoom512_R_acc`
    - `export ALIGNED_TYPE=aligned_N_alpha500_dinov2b_ref`
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s local -t torque -d co3d -r feats/${MESH_FEATURE_TYPE}/${MESH_TYPE}"`  
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s local -t torque -d co3d -r feats_dist/min_avg/${MESH_FEATURE_TYPE}/${MESH_TYPE}"`
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s local -t torque -d co3d -r mesh/${ALIGNED_TYPE}"`  
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s local -t torque -d co3d -r mesh/${ALIGNED_TYPE}_filtered"`    
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s local -t torque -d co3d -r tform_obj/${ALIGNED_TYPE}"`    
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s local -t torque -d co3d -r tform_obj/${ALIGNED_TYPE}_filtered"`    
   
   
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s torque -t local -d co3d -r tform_obj/label3d"` 
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s torque -t local -d co3d -r tform_obj/label3d_cuboid"`
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s torque -t local -d co3d -r mesh/cuboid500"` 

    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s torque -t local -d objectnet3d"`
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s torque -t local -d pascal3d"`  


### Benchmark

To evaluate one method use

  - `od3d bench multiple -b co3d_nemo -p slurm`.

You can evaluate multiple methods, by specyfing an ablation directory, e.g. `nemo_old`

  - `od3d bench multiple -b co3d_nemo -p slurm -a nemo_old`.

To see the current status on slurm use

  - `od3d bench status-slurm`.

To stop a job running on slurm use

  - `od3d bench stop-slurm -j <job-name>`.

### Media
`od3d dataset save-sequences-as-video -d co3d_no_zsp_aligned_visual`    
`od3d dataset visualize-category-sequences -d co3d_no_zsp_aligned_visual`    
`od3d dataset visualize-category-meshes -d co3d_no_zsp_aligned_visual`    
`od3d dataset visualize-category-pcls -d co3d_no_zsp_aligned_visual`  

### Tables
`od3d table multiple -b co3d_aligned_nemo -a categories/cross,nemo_aligned/ref -m test/pascal3d_test/pose/acc_pi6 -r categories/cross -l 24`

### Figures
`od3d figure multiple -a nemo3d_align/dataset,nemo3d_align/mesh_type,nemo3d_align/dist_app_weight,nemo3d_align/dist_cyclic_temp -c method.nemo.geo_cyclic_weight_temp,method.nemo.dist_appear_weight -m pose/acc_pi6,pose/acc_pi18`    

### Documentation
- [Coordinate Frames](docs/coordinate_frames/README.md)