import logging

import torchvision.io.image

logger = logging.getLogger(__name__)
import shutil
from od3d.datasets.co3d import CO3D
from od3d.datasets.co3d.enum import CO3D_CATEGORIES, ALLOW_LIST_FRAME_TYPES, PCL_SOURCES
from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_DATASET_SPLITS
from od3d.cv.io import write_pts3d_with_colors, read_pts3d, read_pts3d_colors
from omegaconf import DictConfig
from pathlib import Path
from od3d.io import run_cmd
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta
from od3d.datasets.co3d.sequence import CO3D_Sequence, CO3D_SequenceMeta
from tqdm import tqdm
from typing import List
from od3d.datasets.object import OD3D_TFROM_OBJ_TYPES

class CO3Dv1_Frame(CO3D_Frame):
    def __post_init__(self):
        # hack: prevents circular import
        self.sequence_type = CO3Dv1_Sequence

class CO3Dv1_Sequence(CO3D_Sequence):

    def __post_init__(self):
        # hack: override frame_type otherwise set in CO3D_Sequence
        self.frame_type = CO3Dv1_Frame

    def get_min_HW(self):
        H = 999999999
        W = 999999999
        for f_id in range(len(self.frames_names)):
            frame = self.get_frame_by_index(f_id)
            frame_H = frame.H
            frame_W = frame.W
            if frame_H < H:
                H = frame_H
            if frame_W < W:
                W = frame_W
        return H, W

    def get_sfm_HW(self):
        H = 999999999
        W = 999999999
        for f_id in range(len(self.frames_names)):
            frame = self.get_frame_by_index(f_id)
            frame_H = frame.H - frame.H % 8  # 8 required fore droid slam to work
            frame_W = frame.W - frame.W % 8  # 8 required fore droid slam to work
            if frame_H < H:
                H = frame_H
            if frame_W < W:
                W = frame_W
        return H, W

from typing import Dict
from od3d.datasets.object import OD3D_MESH_TYPES, OD3D_MESH_FEATS_TYPES, OD3D_MESH_FEATS_DIST_REDUCE_TYPES
class CO3Dv1(CO3D):
    sequence_type = CO3Dv1_Sequence
    frame_type = CO3Dv1_Frame
    tform_obj_type = OD3D_TFROM_OBJ_TYPES.LABEL3D_ZSP_CUBOID

    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[CO3D_CATEGORIES]=None,
                 dict_nested_frames: Dict[str, Dict[str, List[str]]]=None,
                 dict_nested_frames_ban: Dict[str, Dict[str, List[str]]]=None,
                 frames_count_max_per_sequence=None, sequences_count_max_per_category=None,
                 transform=None, index_shift=0, subset_fraction=1.,
                 mesh_type=OD3D_MESH_TYPES.CUBOID500,
                 mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
                 mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
                 tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_ZSP_CUBOID):

        super().__init__(categories=categories, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, subset_fraction=subset_fraction,
                         index_shift=index_shift, dict_nested_frames=dict_nested_frames,
                         dict_nested_frames_ban=dict_nested_frames_ban,
                         frames_count_max_per_sequence=frames_count_max_per_sequence,
                         sequences_count_max_per_category=sequences_count_max_per_category,
                         mesh_type=mesh_type, mesh_feats_type=mesh_feats_type,
                         mesh_feats_dist_reduce_type=mesh_feats_dist_reduce_type, tform_obj_type=tform_obj_type)

    @staticmethod
    def setup(config: DictConfig):

        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous CO3Dv1")
            shutil.rmtree(path_raw)

        if path_raw.exists() and not config.setup.override:
            logger.info(f"Found CO3D dataset at {path_raw}")
        else:
            path_co3d_repo = path_raw.joinpath('co3d')
            path_co3d_repo.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloning CO3D github repository to {path_co3d_repo}")
            run_cmd(cmd=f'cd {path_raw} && git clone git@github.com:facebookresearch/co3d.git', live=True, logger=logger)
            run_cmd(cmd=f'cd {path_raw}/co3d && git fetch', live=True, logger=logger)
            run_cmd(cmd=f'cd {path_raw}/co3d && git checkout v1', live=True, logger=logger)

            logger.info(f"Downloading CO3D dataset at {path_raw}")
            run_cmd(cmd=f'python {path_co3d_repo.joinpath("download_dataset.py")} --download_folder {path_raw}', live=True, logger=logger)

    @staticmethod
    def extract_meta(config: DictConfig):
        path = Path(config.path_raw)
        path_meta = CO3D.get_path_meta(config=config)

        dict_nested_frames = config.get('dict_nested_frames', None)
        dict_nested_frames_banned = config.get('dict_nested_frames_ban', None)
        preprocess_meta_override = config.get('extract_meta', False).get('override', False)
        preprocess_meta_remove_previous = config.get('extract_meta', False).get('remove_previous', False)

        categories = list(dict_nested_frames.keys()) if dict_nested_frames is not None else CO3D_CATEGORIES.list()
        sequences_count_max_per_category = config.get("sequences_count_max_per_category", None)

        if preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        for category in categories:
            logger.info(f'preprocess meta for class {category}')
            sequence_annotations = load_dataclass_jgzip(
                f"{path}/{category}/sequence_annotations.jgz", List[SequenceAnnotation]
            )
            sequences_names = list(dict_nested_frames[category].keys()) if dict_nested_frames is not None and category in dict_nested_frames.keys() and dict_nested_frames[category] is not None else None
            if sequences_names is None and (dict_nested_frames is None or (category in dict_nested_frames.keys() and dict_nested_frames[category] is None)):
                sequences_names = [sequence_annoation.sequence_name for sequence_annoation in tqdm(sequence_annotations)]
            if dict_nested_frames_banned is not None and category in dict_nested_frames_banned.keys() and dict_nested_frames_banned[category] is not None:
                sequences_names = list(filter(lambda seq: seq not in dict_nested_frames_banned[category].keys(), sequences_names))

            logger.info('reading sequence annotations...')
            seq_count_per_class = 0
            read_sequences = []
            for sequence_annoation in tqdm(sequence_annotations):
                if sequences_names is not None and sequence_annoation.sequence_name not in sequences_names:
                    continue

                read_sequences.append(sequence_annoation.sequence_name)
                seq_count_per_class += 1
                if sequences_count_max_per_category is not None:
                    if seq_count_per_class > sequences_count_max_per_category:
                        break

                sequence_name = str(sequence_annoation.sequence_name)
                sequence_meta_fpath = CO3D_SequenceMeta.get_fpath_sequence_meta_with_category_and_name(path_meta=path_meta,
                                                                                                       category=category,
                                                                                                       name=sequence_name)

                if sequence_meta_fpath.exists() and not preprocess_meta_override:
                    continue

                sequence_meta = CO3D_SequenceMeta.load_from_raw(sequence_annotation=sequence_annoation)
                sequence_meta.save(path_meta=path_meta)

            cls_frame_annotations = load_dataclass_jgzip(
                f"{path}/{category}/frame_annotations.jgz", List[FrameAnnotation]
            )
            #cls_frame_annotations = [fa for fa in cls_frame_annotations if fa.meta[
            #    'frame_type'] in ALLOW_LIST_FRAME_TYPES]

            logger.info('reading frame annotations...')
            for frame_annotation in tqdm(cls_frame_annotations):
                if sequences_names is not None and frame_annotation.sequence_name not in sequences_names:
                    continue

                if frame_annotation.sequence_name not in read_sequences:
                    continue

                frame_name = str(frame_annotation.frame_number)
                sequence_name = str(frame_annotation.sequence_name)
                frame_meta_fpath = CO3D_FrameMeta.get_fpath_frame_meta_with_category_sequence_and_frame_name(path_meta=path_meta,
                                                                                                             category=category,
                                                                                                             sequence_name=sequence_name,
                                                                                                             name=frame_name)
                if frame_meta_fpath.exists() and not preprocess_meta_override:
                    continue


                frame_meta = CO3D_FrameMeta.load_from_raw(frame_annotation=frame_annotation)
                frame_meta.save(path_meta=path_meta)


    #
    # def get_sequence_by_category_and_name(self, category, name):
    #     sequence_meta = CO3D_SequenceMeta.load_from_meta_with_category_and_name(path_meta=self.path_meta, category=category, name=name)
    #     return CO3Dv1_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
    #                            meta=sequence_meta, modalities=self.modalities, categories=self.categories,
    #                            mesh_feats_type=self.mesh_feats_type, dist_verts_mesh_feats_reduce_type=self.dist_verts_mesh_feats_reduce_type, cuboid_source=self.cuboid_source,
    #                            cam_tform_obj_source=self.cam_tform_obj_source, pcl_source=self.pcl_source,
    #                            aligned_name=self.aligned_name, mesh_name=self.mesh_name)
    #
    # def get_frame_by_meta(self, frame_meta: CO3D_FrameMeta):
    #     return CO3Dv1_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
    #                       meta=frame_meta, modalities=self.modalities, categories=self.categories,
    #                       cuboid_source=self.cuboid_source, cam_tform_obj_source=self.cam_tform_obj_source,
    #                       aligned_name=self.aligned_name, mesh_name=self.mesh_name, pcl_source=self.pcl_source)


    # def preprocess_sfm(self, override=False):
    #     if not override and self.path_sfm.exists():
    #         logger.info(f'path sfm already exists at {self.path_sfm}')
    #         return
    #     else:
    #         logger.info(f'preprocessing sfm for {self.name_unique} with type {self.sfm_type}')
    #
    #     if self.sfm_type == OD3D_SEQUENCE_SFM_TYPES.DROID:
    #
    #         path_out_root = self.path_sfm_root #  self.path_preprocess.joinpath('droid_slam')
    #         rpath_out = Path(self.name_unique)
    #
    #         path_out = path_out_root.joinpath(rpath_out)
    #         path_in = path_out.joinpath('images')
    #
    #         H = 999999999
    #         W = 999999999
    #         for f_id in range(len(self.frames_names)):
    #             frame = self.get_frame_by_index(f_id)
    #             frame_H = frame.H - frame.H % 8  # 8 required fore droid slam to work
    #             frame_W = frame.W - frame.W % 8  # 8 required fore droid slam to work
    #             if frame_H < H:
    #                 H = frame_H
    #             if frame_W < W:
    #                 W = frame_W
    #
    #         for f_id in range(len(self.frames_names)):
    #             frame = self.get_frame_by_index(f_id)
    #             rgb = frame.rgb[:, :H, :W].clone()
    #             torchvision.io.image.write_jpeg(rgb, filename=str(path_in.joinpath(f'{f_id:05d}' + '.jpg')))
    #
    #         #from od3d.models.model import OD3D_Model
    #         #from od3d.cv.transforms.transform import OD3D_Transform
    #         #from od3d.cv.transforms.sequential import SequentialTransform
    #         #model = OD3D_Model.create_by_name('sam')
    #         #model.cuda()
    #         #model.eval()
    #         #transform = SequentialTransform([OD3D_Transform.create_by_name(''), model.transform])
    #
    #         from od3d.cv.reconstruction.droid_slam import run_droid_slam
    #         run_droid_slam(path_rgbs=path_in, path_out_root=path_out_root, rpath_out=rpath_out,
    #                        cam_intr4x4=self.first_frame.get_cam_intr4x4(), pcl_fname=self.fname_sfm_pcl,
    #                        rays_center3d_fname=self.fname_sfm_rays_center3d,
    #                        cam_tform_obj_dname=self.dname_sfm_cams_tform4x4_obj )
    #     else:
    #         raise NotImplementedError(f'sfm_type {self.sfm_type} not implemented')
    #
    #
    # def preprocess_pcl(self, override=False):
    #     from od3d.datasets.object import OD3D_SEQUENCE_SFM_TYPES, OD3D_PCL_TYPES, OD3D_TFROM_OBJ_TYPES
    #     from od3d.cv.io import get_default_device
    #     from od3d.cv.reconstruction.clean import get_pcl_clean_with_masks
    #     from od3d.cv.io import write_pts3d_with_colors_and_normals
    #
    #     if self.pcl_type == OD3D_PCL_TYPES.META:
    #         logger.info('no need to preprocess pcl for meta pcl type')
    #         return
    #     elif self.pcl_type == OD3D_PCL_TYPES.SFM:
    #         logger.info('no need to preprocess pcl for sfm pcl type')
    #         return
    #     elif self.pcl_type == OD3D_PCL_TYPES.SFM_MASK or self.pcl_type == OD3D_PCL_TYPES.META_MASK:
    #         if self.pcl_type == OD3D_PCL_TYPES.SFM_MASK:
    #             pcl_type_in = OD3D_PCL_TYPES.SFM
    #         elif self.pcl_type == OD3D_PCL_TYPES.META_MASK:
    #             pcl_type_in = OD3D_PCL_TYPES.META
    #         else:
    #             raise NotImplementedError
    #
    #         fpath_pcl_out = self.get_fpath_pcl(pcl_type=self.pcl_type)
    #
    #         if not override and fpath_pcl_out.exists():
    #             logger.info(f'fpath sfm mask pcl already exists at {fpath_pcl_out}')
    #             return
    #
    #         frames = self.get_frames()
    #         device = get_default_device()
    #         H = min([frame.H for frame in frames])
    #         W = min([frame.W for frame in frames])
    #         masks = torch.stack([frame.get_mask()[:, :H, :W] for frame in frames], dim=0).to(device=device)
    #         cams_intr4x4 = torch.stack([frame.read_cam_intr4x4() for frame in frames], dim=0).to(device=device)
    #         cams_tform4x4_obj = torch.stack([frame.read_cam_tform4x4_obj(tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW) for frame in frames], dim=0).to(device=device)
    #
    #         pts3d, pts3d_colors, pts3d_normals = self.read_pcl(pcl_type=pcl_type_in, device=device, tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW)
    #         pts3d, pts3d_mask = get_pcl_clean_with_masks(pcl=pts3d, masks=masks,
    #                                                      cams_intr4x4=cams_intr4x4,
    #                                                      cams_tform4x4_obj=cams_tform4x4_obj,
    #                                                      pts3d_prob_thresh=0.6,
    #                                                      pts3d_max_count=20000,
    #                                                      pts3d_count_min=10,
    #                                                      return_mask=True)
    #         pts3d_colors = pts3d_colors[pts3d_mask]
    #         pts3d_normals = pts3d_normals[pts3d_mask]
    #
    #         write_pts3d_with_colors_and_normals(fpath=fpath_pcl_out,
    #                                             pts3d=pts3d.detach().cpu(),
    #                                             pts3d_colors=pts3d_colors.detach().cpu(),
    #                                             pts3d_normals=pts3d_normals.detach().cpu())

    #
    #
    # def run_droid_slam(self):
    #     path_out_root = self.path_preprocess.joinpath('droid_slam')
    #     rpath_out = self.name_unique
    #     path_out = path_out_root.joinpath(rpath_out)
    #     path_in = path_out.joinpath('images')
    #     if not path_in.exists():
    #         path_in.mkdir(parents=True, exist_ok=True)
    #
    #     H = 999999999
    #     W = 999999999
    #     for f_id in range(len(self.frames_names)):
    #         frame = self.get_frame_by_index(f_id)
    #         frame_H = frame.H - frame.H % 8  # 8 required fore droid slam to work
    #         frame_W = frame.W - frame.W % 8  # 8 required fore droid slam to work
    #         if frame_H < H:
    #             H = frame_H
    #         if frame_W < W:
    #             W = frame_W
    #
    #     for f_id in range(len(self.frames_names)):
    #         frame = self.get_frame_by_index(f_id)
    #         rgb = frame.rgb[:, :H, :W].clone()
    #         torchvision.io.image.write_jpeg(rgb, filename=str(path_in.joinpath(f'{f_id:05d}' + '.jpg')))
    #
    #     from od3d.cv.reconstruction.droid_slam import run_droid_slam
    #     run_droid_slam(path_rgbs=path_in, path_out_root=path_out_root, rpath_out=rpath_out,
    #                    cam_intr4x4=self.first_frame.get_cam_intr4x4(), pcl_fname=self.fname_sfm_pcl,
    #                    rays_center3d_fname=self.fname_sfm_rays_center3d,
    #                    cam_tform_obj_dname=self.dname_sfm_cams_tform4x4_obj)
    #
        #
        # fx = self.first_frame.meta.l_cam_intr4x4[0][0]
        # fy = self.first_frame.meta.l_cam_intr4x4[1][1]
        # cx = self.first_frame.meta.l_cam_intr4x4[0][2]
        # cy = self.first_frame.meta.l_cam_intr4x4[1][2]
        # if not path_out.exists():
        #     path_out.mkdir(parents=True, exist_ok=True)
        # run_cmd(cmd=f'echo "{fx} {fy} {cx} {cy}" > {path_out_root}/{rpath_out}/calib.txt', logger=logger)
        # run_cmd(
        #     cmd=f'docker run --user=$(id -u):$(id -g) --gpus all -e RPATH_OUT={rpath_out} -e STRIDE={stride} -v {path_in}:/home/appuser/in -v {path_out_root}:/home/appuser/DROID-SLAM/reconstructions/out -t {image_tag}',
        #     logger=logger, live=True)
        #
        #

    """
    def preprocess_mesh(self):
        import numpy as np
        from od3d.cv.visual.show import show_scene
        from od3d.cv.geometry.fit.rays_center3d import fit_rays_center3d
        from od3d.cv.geometry.fit.plane4d import fit_plane, score_plane4d_fit
        from od3d.cv.geometry.transform import plane4d_to_tform4x4, tform4x4_from_transl3d
        from od3d.cv.optimization.ransac import ransac
        from od3d.cv.geometry.transform import tform4x4_broadcast
        from od3d.cv.geometry.mesh import Mesh
        from functools import partial

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        scene_particles = 2000
        particle_quantile_dist = 0.05


        fpath_droid_slam_pcl = self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'pcl.ply')
        fpath_droid_slam_traj = self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'traj_est.pt')

        if not fpath_droid_slam_pcl.exists() or not fpath_droid_slam_traj.exists():
            self.run_droid_slam()


        # N x 3
        pts3d = read_pts3d(fpath_droid_slam_pcl).to(device=device)
        pts3d_colors = read_pts3d_colors(fpath_droid_slam_pcl).to(device=device)

        # F x 4 x 4
        cams_tform4x4_obj = torch.load(fpath_droid_slam_traj).to(device=device)

        #show_scene(pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])

        pts3d_range = max(pts3d.max(dim=0)[0] - pts3d.min(dim=0)[0]).item()
        scene_size = pts3d_range
        particle_size = torch.cdist(pts3d[:2000], pts3d[:2000]).quantile(q=particle_quantile_dist)

        center3d = fit_rays_center3d(cams_tform4x4_obj=cams_tform4x4_obj)
        center3d_tform4x4_obj = tform4x4_from_transl3d(-center3d)

        cams_tform4x4_obj = tform4x4_broadcast(cams_tform4x4_obj, inv_tform4x4(center3d_tform4x4_obj))
        pts3d = transf3d_broadcast(pts3d, transf4x4=center3d_tform4x4_obj)
        center3d = transf3d_broadcast(center3d, transf4x4=center3d_tform4x4_obj)



        center3d_mesh = Mesh.create_sphere(center3d=center3d, radius=scene_size / 30., device=device)
        #show_scene(meshes=[center3d_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])

        mask_plane_thresh = particle_size / 5.
        cams_traj = inv_tform4x4(cams_tform4x4_obj)[:, :3, 3]
        plane4d = ransac(pts=pts3d, fit_func=fit_plane,
                         score_func=partial(score_plane4d_fit, plane_dist_thresh=mask_plane_thresh,
                                            cams_traj=cams_traj), fits_count=1000, fit_pts_count=3)


        plane3d_tform4x4_obj = plane4d_to_tform4x4(plane4d)


        #plane_z = -plane3d_tform4x4_obj[2, 3]
        #plane3d_tform4x4_obj[:3, 3] = 0.
        cams_tform4x4_obj = tform4x4_broadcast(cams_tform4x4_obj, inv_tform4x4(plane3d_tform4x4_obj))
        pts3d = transf3d_broadcast(pts3d, transf4x4=plane3d_tform4x4_obj)
        center3d = transf3d_broadcast(center3d, transf4x4=plane3d_tform4x4_obj)

        height = scene_size / 5.
        radius = scene_size * 0.5
        plan3d_mesh = Mesh.create_plane_as_cone(center3d, radius=radius, height=height, device=device)

        # show_scene(meshes=[center3d_mesh, plan3d_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])

        #plane_tform4x4_pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=plane3d_tform4x4_obj)
        mask_pts3d_on_plane = pts3d[:, 2] < mask_plane_thresh
        # starting with 10 percentage of points
        mask_center_thresh = (pts3d[~mask_pts3d_on_plane] - center3d).norm(dim=-1).quantile(0.1)
        mask_pts3d_obj = (~mask_pts3d_on_plane) * ((pts3d - center3d).norm(dim=-1) < mask_center_thresh)

        pts3d_not_on_plane = pts3d[~mask_pts3d_on_plane]
        dists_pts3d_not_on_plane_plane = pts3d[~mask_pts3d_on_plane, 2]
        dists_pts3d_not_on_plane_obj_dists = (pts3d[mask_pts3d_obj][:, None] - pts3d[~mask_pts3d_on_plane][None, :]).norm(dim=-1).min(dim=0)[0]

        while ((dists_pts3d_not_on_plane_obj_dists < particle_size) * (dists_pts3d_not_on_plane_obj_dists < dists_pts3d_not_on_plane_plane)).sum() > mask_pts3d_obj.sum():
            logger.info(mask_pts3d_obj.sum())
            mask_pts3d_obj[~mask_pts3d_on_plane] += (dists_pts3d_not_on_plane_obj_dists < particle_size) * (dists_pts3d_not_on_plane_obj_dists < dists_pts3d_not_on_plane_plane)
            dists_pts3d_not_on_plane_obj_dists = torch.cdist(pts3d[mask_pts3d_obj], pts3d_not_on_plane).min(dim=0)[0]

        pts3d_obj = pts3d[mask_pts3d_obj]

        pts3d_colors[mask_pts3d_on_plane] = torch.Tensor([0., 0., 1.]).to(device=device)
        pts3d_colors[mask_pts3d_obj] = torch.Tensor([0., 1., 0.]).to(device=device)
        pts3d_colors[(~mask_pts3d_on_plane) * (~mask_pts3d_obj)] = torch.Tensor([1., 0., 0.]).to(device=device)

        # show_scene(meshes=[center3d_mesh, plan3d_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])

        o3d_pcl = open3d.geometry.PointCloud()
        o3d_pcl.points = open3d.utility.Vector3dVector(pts3d_obj[:].detach().cpu().numpy())
        o3d_pcl.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        #o3d_pcl.estimate_normals()

        alpha = mask_center_thresh
        #mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcl, depth=9)
        # o3d_obj_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcl, alpha)
        o3d_obj_mesh, _ = o3d_pcl.compute_convex_hull()
        o3d_obj_mesh.compute_vertex_normals()

        #o3d_obj_mesh = o3d_obj_mesh.subdivide_midpoint(number_of_iterations=2)
        o3d_obj_mesh = o3d_obj_mesh.subdivide_loop(number_of_iterations=2)

        o3d_obj_mesh = o3d_obj_mesh.simplify_vertex_clustering(voxel_size=o3d_obj_mesh.get_volume() / 1000.)

        #o3d_obj_mesh.get_volume()
        #o3d_obj_mesh.get_surface_area()

        obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)
        show_scene(meshes=[obj_mesh, center3d_mesh, plan3d_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors],
                   cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4], meshes_as_wireframe=True)

        logger.info(self.name_unique)
        # open3d.visualization.draw(geometries)
        obj_tform_droid_slam = tform4x4(plane3d_tform4x4_obj, center3d_tform4x4_obj)
        # save point cloud clean
        pts3d = read_pts3d(fpath_droid_slam_pcl).to(device=device)
        pts3d = transf3d_broadcast(pts3d, transf4x4=obj_tform_droid_slam)
        pts3d_colors = read_pts3d_colors(fpath_droid_slam_pcl).to(device=device)

        # note: only if mask is not available due to downsampling consider this
        #vertices_dist_median = torch.cdist(pts3d_obj[None,], pts3d_obj[None,])[0].median()
        #mask_pts3d_obj = torch.cdist(pts3d[None,], pts3d_obj[None,])[0].min(dim=-1).values < vertices_dist_median / 5.

        pts3d_clean = pts3d[mask_pts3d_obj]
        pts3d_colors_clean = pts3d_colors[mask_pts3d_obj]

        self.fpath_pcl_droid_slam_clean.parent.mkdir(parents=True, exist_ok=True)
        self.fpath_pcl_droid_slam.parent.mkdir(parents=True, exist_ok=True)
        write_pts3d_with_colors(fpath=self.fpath_pcl_droid_slam_clean, pts3d=pts3d_clean,
                                pts3d_colors=pts3d_colors_clean)
        write_pts3d_with_colors(fpath=self.fpath_pcl_droid_slam, pts3d=pts3d, pts3d_colors=pts3d_colors)

        obj_mesh.write_to_file(fpath=self.fpath_mesh)

        # save transformation
        for i, cam_tform4x4_obj in enumerate(cams_tform4x4_obj):
            frame = self.get_frame_by_index(i)
            frame.fpath_cam_tform4x4_obj_droid_slam.parent.mkdir(parents=True, exist_ok=True)
            torch.save(obj=cam_tform4x4_obj.detach().cpu(), f=frame.fpath_cam_tform4x4_obj_droid_slam)    
        """
