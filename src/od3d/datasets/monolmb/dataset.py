import logging
logger = logging.getLogger(__name__)

from od3d.datasets.monolmb.frame import MonoLMB_Frame, MonoLMB_FrameMeta
from od3d.datasets.monolmb.sequence import MonoLMB_Sequence

from od3d.datasets.dataset import OD3D_Dataset, OD3D_SequenceDataset
from od3d.datasets.object import OD3D_FRAME_MASK_TYPES, OD3D_CAM_TFORM_OBJ_TYPES, OD3D_MESH_TYPES, OD3D_PCL_TYPES, \
    OD3D_SEQUENCE_SFM_TYPES, OD3D_TFROM_OBJ_TYPES, OD3D_MESH_FEATS_TYPES, OD3D_MESH_FEATS_DIST_REDUCE_TYPES
from od3d.datasets.sequence_meta import OD3D_SequenceMetaCategoryMixin

from pathlib import Path
from omegaconf import DictConfig
import shutil

class MonoLMB(OD3D_SequenceDataset):
    from od3d.datasets.monolmb.enum import MONOLMB_CATEGORIES
    all_categories = list(MONOLMB_CATEGORIES)
    sequence_type = MonoLMB_Sequence # od3d.datasets.monolmb.sequence.MonoLMB_Sequence
    frame_type = MonoLMB_Frame # od3d.datasets.monolmb.frame.MonoLMB_Frame

    def path_videos(self):
        return MonoLMB.get_path_videos(self.path_raw)

    @staticmethod
    def get_path_videos(path_raw: Path):
        return path_raw.joinpath('videos')

    def path_frames_rgb(self):
        return MonoLMB.get_path_frames_rgb(path_raw=self.path_raw)

    @staticmethod
    def get_path_frames_rgb(path_raw: Path):
        return path_raw.joinpath('frames')

    def get_frame_by_name_unique(self, name_unique):
        return self.frame_type(path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                               name_unique=name_unique, all_categories=self.categories,
                               mask_type=OD3D_FRAME_MASK_TYPES.SAM_SFM_RAYS_CENTER3D,
                               cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.SFM,
                               mesh_type=OD3D_MESH_TYPES.CUBOID500,
                               mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
                               mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
                               pcl_type=OD3D_PCL_TYPES.SFM_MASK,
                               sfm_type=OD3D_SEQUENCE_SFM_TYPES.DROID,
                               modalities=self.modalities,
                               tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D,)

    def get_sequence_by_name_unique(self, name_unique):
        return self.sequence_type(name_unique=name_unique,
                                  path_raw=self.path_raw,
                                  path_preprocess=self.path_preprocess,
                                  all_categories=self.categories,
                                  mask_type=OD3D_FRAME_MASK_TYPES.SAM_SFM_RAYS_CENTER3D,
                                  cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.SFM,
                                  mesh_type=OD3D_MESH_TYPES.CUBOID500,
                                  mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
                                  mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
                                  pcl_type=OD3D_PCL_TYPES.SFM_MASK,
                                  sfm_type=OD3D_SEQUENCE_SFM_TYPES.DROID,
                                  modalities=self.modalities,
                                  tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D,)

    @staticmethod
    def setup(config: DictConfig):
        logger.info('recording video...')

        # /misc/lmbraid19/sommerl/datasets/MonoLMB/videos/elephant/24_01_29__18_10.mp4
        #
        from od3d.cv.io import extract_frames_from_video

        #for fpath_video in fpath_videos
        path_videos = Path('/misc/lmbraid19/sommerl/datasets/MonoLMB/videos')
        path_frames = Path('/misc/lmbraid19/sommerl/datasets/MonoLMB/frames')
        fpaths_videos = [file for file in path_videos.rglob('*') if file.is_file()]

        for fpath_video in fpaths_videos:
            path_video_frames = path_frames.joinpath(fpath_video.relative_to(path_videos).with_suffix(""))
            extract_frames_from_video(fpath_video, path_video_frames, fps=5)

    @staticmethod
    def extract_meta(config: DictConfig):
        path_meta = MonoLMB.get_path_meta(config=config)
        path_raw = Path(config.path_raw)
        path_sequences = MonoLMB.get_path_frames_rgb(path_raw=path_raw)
        config.setup.remove_previous = True
        if path_meta.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous MonoLMB meta")
            shutil.rmtree(path_meta)

        path_meta.mkdir(parents=True, exist_ok=True)

        dict_nested_frames = config.get('dict_nested_frames', None)
        dict_nested_frames_banned = config.get('dict_nested_frames_ban', None)
        preprocess_meta_override = config.get('extract_meta', False).get('override', False)

        from od3d.datasets.monolmb.enum import MONOLMB_CATEGORIES
        categories = list(dict_nested_frames.keys()) if dict_nested_frames is not None else MONOLMB_CATEGORIES.list()
        sequences_count_max_per_category = config.get("sequences_count_max_per_category", None)

        for category in categories:
            logger.info(f'preprocess meta for class {category}')

            sequences_names = list(dict_nested_frames[category].keys()) if dict_nested_frames is not None and category in dict_nested_frames.keys() and dict_nested_frames[category] is not None else None
            if sequences_names is None and (dict_nested_frames is None or (category in dict_nested_frames.keys() and dict_nested_frames[category] is None)):
                if not path_sequences.joinpath(category).exists():
                    continue

                sequences_names = [fpath.stem for fpath in path_sequences.joinpath(category).iterdir()]
            if dict_nested_frames_banned is not None and category in dict_nested_frames_banned.keys() and dict_nested_frames_banned[category] is not None:
                sequences_names = list(filter(lambda seq: seq not in dict_nested_frames_banned[category].keys(), sequences_names))

            for sequence_name in sequences_names:
                fpath_sequence_meta = OD3D_SequenceMetaCategoryMixin.get_fpath_sequence_meta_with_category_and_name(
                    path_meta=path_meta, category=category, name=sequence_name)

                if fpath_sequence_meta.exists() and not preprocess_meta_override:
                    continue

                sequence_meta = OD3D_SequenceMetaCategoryMixin.load_from_raw(category=category, name=sequence_name)

                if len(list(path_sequences.joinpath(sequence_meta.name_unique).iterdir())) > 0:
                    # filtering sequences with no rgb images
                    sequence_meta.save(path_meta=path_meta)

                for fpath_frame in path_sequences.joinpath(sequence_meta.name_unique).iterdir():
                    import torch
                    from od3d.cv.io import read_image
                    H, W = read_image(fpath_frame).shape[1:]
                    l_size = torch.LongTensor([H, W]).tolist()
                    frame_meta = MonoLMB_FrameMeta.load_from_raw(
                        name=fpath_frame.stem, category=category, sequence_name=sequence_name,
                        rfpath_rgb=Path(fpath_frame.relative_to(path_raw)), l_size=l_size)


                    frame_meta.save(path_meta=path_meta)

