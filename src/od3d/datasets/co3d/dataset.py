import logging
logger = logging.getLogger(__name__)

from od3d.datasets.frame import OD3D_FRAME_MODALITIES
from od3d.datasets.co3d.enum import CO3D_CATEGORIES
from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta
from od3d.datasets.co3d.sequence import CO3D_Sequence
from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES, CUBOID_SOURCES, CO3D_FRAME_TYPES, CO3D_FRAME_SPLITS, CO3D_CATEGORIES, MAP_CATEGORIES_OD3D_TO_CO3D, FEATURE_TYPES, REDUCE_TYPES, ALLOW_LIST_FRAME_TYPES, PCL_SOURCES

from od3d.datasets.dataset import OD3D_Dataset, OD3D_SequenceDataset
from od3d.datasets.object import OD3D_FRAME_MASK_TYPES, OD3D_CAM_TFORM_OBJ_TYPES, OD3D_MESH_TYPES, OD3D_PCL_TYPES, \
    OD3D_SEQUENCE_SFM_TYPES, OD3D_TFROM_OBJ_TYPES, OD3D_MESH_FEATS_TYPES, OD3D_MESH_FEATS_DIST_REDUCE_TYPES, OD3D_FRAME_DEPTH_TYPES
from pathlib import Path
from typing import List, Dict

from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta
from od3d.datasets.co3d.sequence import CO3D_Sequence, CO3D_SequenceMeta
from od3d.io import run_cmd
from omegaconf import DictConfig
import shutil
from tqdm import tqdm

class CO3D(OD3D_SequenceDataset):
    all_categories = list(CO3D_CATEGORIES)
    map_od3d_categories = MAP_CATEGORIES_OD3D_TO_CO3D
    sequence_type = CO3D_Sequence
    frame_type = CO3D_Frame
    tform_obj_type = OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID
    sfm_type = OD3D_SEQUENCE_SFM_TYPES.META
    cam_tform_obj_type = OD3D_CAM_TFORM_OBJ_TYPES.META
    mask_type = OD3D_FRAME_MASK_TYPES.META
    pcl_type = OD3D_PCL_TYPES.META_MASK
    mesh_type = OD3D_MESH_TYPES.CUBOID500
    mesh_feats_type = OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC
    mesh_feats_dist_reduce_type = OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG

    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[CO3D_CATEGORIES]=None,
                 dict_nested_frames: Dict[str, Dict[str, List[str]]]=None,
                 dict_nested_frames_ban: Dict[str, Dict[str, List[str]]]=None,
                 frames_count_max_per_sequence=None, sequences_count_max_per_category=None,
                 transform=None, index_shift=0, subset_fraction=1.,
                 mesh_type=OD3D_MESH_TYPES.CUBOID500,
                 mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
                 mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
                 tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID):

        super().__init__(categories=categories, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, subset_fraction=subset_fraction,
                         index_shift=index_shift, dict_nested_frames=dict_nested_frames,
                         dict_nested_frames_ban=dict_nested_frames_ban,
                         frames_count_max_per_sequence=frames_count_max_per_sequence,
                         sequences_count_max_per_category=sequences_count_max_per_category)

        self.mesh_type = mesh_type
        self.mesh_feats_type = mesh_feats_type
        self.mesh_feats_dist_reduce_type = mesh_feats_dist_reduce_type
        self.tform_obj_type = tform_obj_type

    def get_frame_by_name_unique(self, name_unique):
        return self.frame_type(path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                               name_unique=name_unique, all_categories=self.categories,
                               mask_type=self.mask_type,
                               cam_tform4x4_obj_type=self.cam_tform_obj_type,
                               mesh_type=self.mesh_type,
                               mesh_feats_type=self.mesh_feats_type,
                               mesh_feats_dist_reduce_type=self.mesh_feats_dist_reduce_type,
                               pcl_type=self.pcl_type,
                               sfm_type=self.sfm_type,
                               modalities=self.modalities,
                               tform_obj_type=self.tform_obj_type,
                               depth_type=OD3D_FRAME_DEPTH_TYPES.META)

    def get_sequence_by_name_unique(self, name_unique):
        return self.sequence_type(name_unique=name_unique,
                                  path_raw=self.path_raw,
                                  path_preprocess=self.path_preprocess,
                                  all_categories=self.categories,
                                  mask_type=self.mask_type,
                                  cam_tform4x4_obj_type=self.cam_tform_obj_type,
                                  mesh_type=self.mesh_type,
                                  mesh_feats_type=self.mesh_feats_type,
                                  mesh_feats_dist_reduce_type=self.mesh_feats_dist_reduce_type,
                                  pcl_type=self.pcl_type,
                                  sfm_type=self.sfm_type,
                                  modalities=self.modalities,
                                  tform_obj_type=self.tform_obj_type,
                                  depth_type=OD3D_FRAME_DEPTH_TYPES.META)

    @staticmethod
    def setup(config: DictConfig):

        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous CO3D")
            shutil.rmtree(path_raw)

        if path_raw.exists() and not config.setup.override:
            logger.info(f"Found CO3D dataset at {path_raw}")
        else:
            path_co3d_repo = path_raw.joinpath('co3d')
            path_co3d_repo.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloning CO3D github repository to {path_co3d_repo}")
            run_cmd(cmd=f'cd {path_raw} && git clone git@github.com:facebookresearch/co3d.git', live=True, logger=logger)
            logger.info(f"Downloading CO3D dataset at {path_raw}")
            run_cmd(cmd=f'python {path_co3d_repo.joinpath("co3d/download_dataset.py")} --download_folder {path_raw}', live=True, logger=logger)
            # --n_download_workers 1 --n_extract_workers 1

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

                if len(list(path.joinpath(sequence_meta.name_unique, "images").iterdir())) > 0:
                    # filtering sequences with no rgb images
                    sequence_meta.save(path_meta=path_meta)

            cls_frame_annotations = load_dataclass_jgzip(
                f"{path}/{category}/frame_annotations.jgz", List[FrameAnnotation]
            )
            cls_frame_annotations = [fa for fa in cls_frame_annotations if fa.meta[
                'frame_type'] in ALLOW_LIST_FRAME_TYPES]

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
# from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_DATASET_SPLITS
# from omegaconf import DictConfig
# from co3d.dataset.data_types import (
#     load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
# )
# from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta
# from od3d.datasets.co3d.sequence import CO3D_Sequence, CO3D_SequenceMeta
# import random
#
# from typing import List, Dict, Tuple
# import torch
# from pathlib import Path
#
# from enum import Enum
# from od3d.io import run_cmd
# from copy import copy
#
# import logging
# logger = logging.getLogger(__name__)
# import shutil
# from tqdm import tqdm
# import torch.utils.data
# from od3d.cv.io import load_ply, save_ply
#
# from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES, CUBOID_SOURCES, CO3D_FRAME_TYPES, CO3D_FRAME_SPLITS, CO3D_CATEGORIES, MAP_CATEGORIES_OD3D_TO_CO3D, FEATURE_TYPES, REDUCE_TYPES, ALLOW_LIST_FRAME_TYPES, PCL_SOURCES
#
#
# class CO3D(OD3D_Dataset):
#
#     CATEGORIES = CO3D_CATEGORIES
#     MAP_OD3D_CATEGORIES = MAP_CATEGORIES_OD3D_TO_CO3D
#
#     def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
#                  categories: List[CO3D_CATEGORIES]=None,
#                  dict_nested_frames: Dict[str, Dict[str, List[str]]]=None,
#                  dict_nested_frames_ban: Dict[str, Dict[str, List[str]]]=None,
#                  frames_block_negative_depth=False,
#                  frames_count_max_per_sequence=None,
#                  sequences_require_pcl=False,
#                  sequences_sort_pcl_score=False,
#                  sequences_require_pcl_score=-1000.1,
#                  sequences_require_gt_pose=False,
#                  sequences_require_good_cam_movement=False,
#                  sequences_require_no_missing_frames=True,
#                  sequences_count_max_per_category=None,
#                  sequences_require_mesh=False,
#                  cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D.value,
#                  cuboid_source=CUBOID_SOURCES.DEFAULT.value,
#                  mesh_feats_type=FEATURE_TYPES.DINOV2_AVG.value,
#                  dist_verts_mesh_feats_reduce_type=REDUCE_TYPES.MIN.value,
#                  pcl_source=PCL_SOURCES.CO3D.value,
#                  transform=None, index_shift=0, subset_fraction=1.,
#                  mesh_name: str='default',
#                  aligned_name: str = None):
#
#         if categories is not None:
#             categories = [self.MAP_OD3D_CATEGORIES[category] if category not in self.CATEGORIES.list() else category for category in categories]
#         else:
#             categories = self.CATEGORIES.list()
#
#         self.categories = categories
#         self.path_raw = Path(path_raw)
#         self.path_preprocess = Path(path_preprocess)
#         self.modalities = modalities
#         self.cam_tform_obj_source = cam_tform_obj_source # required for block negative depth
#         self.mesh_feats_type = mesh_feats_type
#         self.dist_verts_mesh_feats_reduce_type = dist_verts_mesh_feats_reduce_type
#         self.frames_count_max_per_sequence = frames_count_max_per_sequence
#         self.frames_block_negative_depth = frames_block_negative_depth
#         self.cuboid_source = cuboid_source
#         self.aligned_name = aligned_name
#         self.mesh_name = mesh_name
#         self.pcl_source = pcl_source
#
#         logger.info("filtering sequences...")
#         self.dict_category_sequences_names = self.filter_dict_nested_sequences(dict_nested_frames=
#                                                                                dict_nested_frames,
#                                                                                require_pcl=sequences_require_pcl,
#                                                                                sort_pcl_score=sequences_sort_pcl_score,
#                                                                                require_pcl_score=sequences_require_pcl_score,
#                                                                                require_good_cam_movement=sequences_require_good_cam_movement,
#                                                                                require_no_missing_frames=sequences_require_no_missing_frames,
#                                                                                require_gt_pose=sequences_require_gt_pose,
#                                                                                count_max_per_category=sequences_count_max_per_category,
#                                                                                sequences_require_mesh=sequences_require_mesh,
#                                                                                dict_nested_frames_ban=dict_nested_frames_ban)
#
#         logger.info(f'sequences filtered')
#         sequences_filtered_str = '\n'
#         for category in self.dict_category_sequences_names.keys():
#             if len(self.dict_category_sequences_names[category]) > 0:
#                 sequences_filtered_str += category + ': \n'
#                 for sequence_name in self.dict_category_sequences_names[category]:
#                     sequences_filtered_str += f"  '{sequence_name}':\n"
#         logger.info(sequences_filtered_str)
#
#         logger.info(f'Found sequences per category')
#
#         dict_nested_frames_seqs_filtered = {}
#         for category in self.dict_category_sequences_names.keys(): #.keys():
#             logger.info(f'{category}: {len(self.dict_category_sequences_names[category])}')
#             if len(self.dict_category_sequences_names[category]) == 0:
#                 continue
#             if dict_nested_frames is not None and category in dict_nested_frames.keys():
#                 dict_nested_frames_seqs_filtered[category] = dict_nested_frames[category]
#             else:
#                 if dict_nested_frames is None:
#                     dict_nested_frames_seqs_filtered[category] = None
#                 else:
#                     # category not in dict_nested_frames
#                     dict_nested_frames_seqs_filtered[category] = {}
#
#             for sequence_name in self.dict_category_sequences_names[category]:
#                 if dict_nested_frames is not None and category in dict_nested_frames.keys() and dict_nested_frames[category] is not None and sequence_name in dict_nested_frames[category]:
#                     dict_nested_frames_seqs_filtered[category][sequence_name] = dict_nested_frames[category][sequence_name]
#                 else:
#                     if dict_nested_frames is None or (category in dict_nested_frames and dict_nested_frames_seqs_filtered[category] is None):
#                         if not isinstance(dict_nested_frames_seqs_filtered[category], dict):
#                             dict_nested_frames_seqs_filtered[category] = {}
#                         dict_nested_frames_seqs_filtered[category][sequence_name] = None
#                     else:
#                         # category / sequence not in dict_nested_frames
#                         dict_nested_frames_seqs_filtered[category][sequence_name] = []
#         dict_nested_frames = dict_nested_frames_seqs_filtered
#
#         super().__init__(categories=categories, name=name, modalities=modalities, path_raw=path_raw,
#                          path_preprocess=path_preprocess, transform=transform, subset_fraction=subset_fraction,
#                          index_shift=index_shift, dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban)
#
#         self.splits_featured = [OD3D_DATASET_SPLITS.RANDOM, OD3D_DATASET_SPLITS.SEQUENCES_SEPARATED, OD3D_DATASET_SPLITS.SEQUENCES_SHARED]
#
#
#     def get_subset_by_sequences(self, dict_category_sequences: Dict[str, List[str]], frames_count_max_per_sequence=None):
#         dict_nested_frames = {}
#         for cat, seqs in dict_category_sequences.items():
#             dict_nested_frames[cat] = {}
#             for seq in seqs:
#                  dict_nested_frames[cat][seq] = None
#         return CO3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
#                     path_preprocess=self.path_preprocess, categories=self.categories,
#                     frames_count_max_per_sequence=frames_count_max_per_sequence,
#                     dict_nested_frames=dict_nested_frames,
#                     cam_tform_obj_source=self.cam_tform_obj_source,
#                     pcl_source=self.pcl_source,
#                     cuboid_source=self.cuboid_source, transform=self.transform, index_shift=self.index_shift,
#                     aligned_name=self.aligned_name,
#                     mesh_name=self.mesh_name)
#
#     def get_split_sequences_shared(self, fraction1: float):
#         dict_category_sequence_name_frames_names_subsetA = {}
#         dict_category_sequence_name_frames_names_subsetB = {}
#         dict_category_sequence_name_frames_names = CO3D_FrameMeta.rollup_flattened_frames(self.list_frames_unique)
#         #dict_category_sequence_name_frames_names = self.list_categories_sequences_names_frames_names_to_dict(self.list_frames_unique)
#         for category, dict_sequence_name_frames_names in dict_category_sequence_name_frames_names.items():
#             dict_category_sequence_name_frames_names_subsetA[category] = {}
#             dict_category_sequence_name_frames_names_subsetB[category] = {}
#             for sequence_name, frames_names in dict_sequence_name_frames_names.items():
#                 frames_names = sorted(frames_names, key=lambda fn: int(fn))
#                 cutoff = int(len(frames_names) * fraction1)
#                 dict_category_sequence_name_frames_names_subsetA[category][sequence_name] = frames_names[:cutoff]
#                 dict_category_sequence_name_frames_names_subsetB[category][sequence_name] = frames_names[cutoff:]
#
#         return self.get_split_from_dicts(dict_category_sequence_name_frames_names_subsetA, dict_category_sequence_name_frames_names_subsetB)
#
#     def get_split_sequences_separated(self, fraction1: float):
#         dict_category_sequence_name_frames_names_subsetA = {}
#         dict_category_sequence_name_frames_names_subsetB = {}
#         dict_category_sequence_name_frames_names = CO3D_FrameMeta.rollup_flattened_frames(self.list_frames_unique)
#         #dict_category_sequence_name_frames_names = self.list_categories_sequences_names_frames_names_to_dict(self.list_frames_unique)
#         for category, dict_sequence_name_frames_names in dict_category_sequence_name_frames_names.items():
#             seqs_names = list(dict_sequence_name_frames_names.keys())
#             cutoff = int(len(seqs_names) * fraction1)
#             dict_category_sequence_name_frames_names_subsetA[category] = {s: dict_sequence_name_frames_names[s] for s in seqs_names[:cutoff]}
#             dict_category_sequence_name_frames_names_subsetB[category] = {s: dict_sequence_name_frames_names[s] for s in seqs_names[cutoff:]}
#
#         return self.get_split_from_dicts(dict_category_sequence_name_frames_names_subsetA, dict_category_sequence_name_frames_names_subsetB)
#
#     """
#     def get_subset_with_item_ids(self, item_ids):
#         list_categories_sequences_names_frames_names = [self.list_frames_unique[id] for id in item_ids]
#         dict_category_sequence_name_frames_names = self.list_categories_sequences_names_frames_names_to_dict(list_categories_sequences_names_frames_names)
#         return self.get_subset_with_dict_category_sequence_name_frames_names(dict_category_sequence_name_frames_names)
#
#     def get_subset_with_names_unique(self, names_unique: List[str]):
#         dict_category_sequence_name_frames_names = CO3D_FrameMeta.get_dict_category_sequence_name_frames_names_with_names_unique(names_unique=names_unique)
#         return self.get_subset_with_dict_category_sequence_name_frames_names(
#             dict_nested_frames=dict_category_sequence_name_frames_names)
#     """
#
#     def get_subset_with_dict_nested_frames(self, dict_nested_frames):
#         return CO3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
#                     path_preprocess=self.path_preprocess, categories=self.categories,
#                     dict_nested_frames=dict_nested_frames,
#                     cam_tform_obj_source=self.cam_tform_obj_source, pcl_source=self.pcl_source,
#                     cuboid_source=self.cuboid_source, transform=self.transform, index_shift=self.index_shift, aligned_name=self.aligned_name,
#                     mesh_name=self.mesh_name)
#
#     def get_split_from_dicts(self, dict_nested_frames_subsetA, dict_nested_frames_subsetB):
#         co3d_subsetA = CO3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
#                             path_preprocess=self.path_preprocess, categories=self.categories,
#                             dict_nested_frames=dict_nested_frames_subsetA,
#                             cam_tform_obj_source=self.cam_tform_obj_source, pcl_source=self.pcl_source,
#                             cuboid_source=self.cuboid_source, transform=self.transform, index_shift=self.index_shift, aligned_name=self.aligned_name,
#                             mesh_name=self.mesh_name)
#
#         co3d_subsetB = CO3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
#                             path_preprocess=self.path_preprocess, categories=self.categories,
#                             dict_nested_frames=dict_nested_frames_subsetB,
#                             cam_tform_obj_source=self.cam_tform_obj_source, pcl_source=self.pcl_source,
#                             cuboid_source=self.cuboid_source, transform=self.transform, index_shift=self.index_shift, aligned_name=self.aligned_name,
#                             mesh_name=self.mesh_name)
#
#         return co3d_subsetA, co3d_subsetB
#
#     def get_item(self, item):
#         frame_meta = CO3D_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
#         return self.get_frame_by_meta(frame_meta=frame_meta)
#
#     def filter_dict_nested_frames(self, dict_nested_frames: Dict[str, Dict[str, List[str]]]):
#         dict_nested_frames = super().filter_dict_nested_frames(dict_nested_frames=dict_nested_frames)
#
#         frames_count_max_per_sequence = self.frames_count_max_per_sequence
#         block_negative_depth = self.frames_block_negative_depth
#
#         # filter frames to exist in sequences:
#         dict_nested_frames_filtered = {}
#         for category, sequences in self.dict_category_sequences_names.items():
#             dict_nested_frames_filtered[category] = {}
#             for sequence in sequences:
#                 dict_nested_frames_filtered[category][sequence] = dict_nested_frames[category][sequence]
#         dict_nested_frames = dict_nested_frames_filtered
#
#         #if frames_count_max_per_sequence is not None or block_negative_depth:
#         #    dict_nested_frames = CO3D_FrameMeta.complete_nested_metas(path_meta=self.path_meta, dict_nested_metas=dict_nested_frames)
#
#         if frames_count_max_per_sequence is not None:
#             dict_nested_frames_filtered = {}
#             for category, dict_sequence_name_frames_names in dict_nested_frames.items():
#                 for sequence_name, frames_names in dict_sequence_name_frames_names.items():
#                     frames_names_filtered = CO3D_FrameMeta.get_subset_frames_names_uniform(frames_names, count_max_per_sequence=frames_count_max_per_sequence)
#                     if category not in dict_nested_frames_filtered.keys():
#                         dict_nested_frames_filtered[category] = {}
#                     dict_nested_frames_filtered[category][sequence_name] = frames_names_filtered
#             dict_nested_frames = dict_nested_frames_filtered
#
#         if block_negative_depth:
#             dict_nested_frames_filtered = {}
#             for category, dict_sequence_name_frames_names in dict_nested_frames.items():
#                 for sequence_name, frames_names in dict_sequence_name_frames_names.items():
#                     frames: List[CO3D_Frame] = [
#                         self.get_frame_by_category_sequence_and_frame_name(category=category,
#                                                                            sequence_name=sequence_name,
#                                                                            frame_name=frame_name)
#                         for frame_name in frames_names]
#                     frames = list(filter(lambda frame: frame.cam_tform4x4_obj[2, 3] >= 0.01, frames))
#                     dict_nested_frames_filtered[category][sequence_name] = [frame.name for frame in frames]
#         return dict_nested_frames
#
#     def visualize(self, item: int):
#         pass
#
#     def get_frame_by_category_sequence_and_frame_name(self, category, sequence_name, frame_name):
#         frame_meta = CO3D_FrameMeta.load_from_meta_with_category_sequence_and_frame_name(
#             path_meta=self.path_meta,
#             category=category,
#             sequence_name=sequence_name,
#             frame_name=frame_name)
#         return self.get_frame_by_meta(frame_meta=frame_meta)
#
#     def get_frame_by_name_unique(self, name_unique: str):
#         frame_meta = CO3D_FrameMeta.load_from_meta_with_name_unique(
#             path_meta=self.path_meta,
#             name_unique=name_unique)
#         return self.get_frame_by_meta(frame_meta=frame_meta)
#
#     def get_frame_by_meta(self, frame_meta: CO3D_FrameMeta):
#         return CO3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
#                           meta=frame_meta, modalities=self.modalities, categories=self.categories,
#                           cuboid_source=self.cuboid_source, cam_tform_obj_source=self.cam_tform_obj_source,
#                           aligned_name=self.aligned_name, mesh_name=self.mesh_name, pcl_source=self.pcl_source,)
#
#     def get_sequences(self):
#         seqs = []
#         for category in self.dict_category_sequences_names.keys():
#             for sequence_name in self.dict_category_sequences_names[category]:
#                 seqs.append(self.get_sequence_by_category_and_name(category=category, name=sequence_name))
#         return seqs
#
#     def get_sequence_by_category_and_name(self, category, name):
#         sequence_meta = CO3D_SequenceMeta.load_from_meta_with_category_and_name(path_meta=self.path_meta, category=category, name=name)
#         return CO3D_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
#                              meta=sequence_meta, modalities=self.modalities, categories=self.categories, aligned_name=self.aligned_name,
#                              mesh_feats_type=self.mesh_feats_type, dist_verts_mesh_feats_reduce_type=self.dist_verts_mesh_feats_reduce_type, cuboid_source=self.cuboid_source,
#                              cam_tform_obj_source=self.cam_tform_obj_source, pcl_source=self.pcl_source,
#                              mesh_name=self.mesh_name)
#
#     def filter_dict_nested_sequences(self, dict_nested_frames: Dict[str, Dict[str, List[str]]], require_pcl, sort_pcl_score, require_pcl_score, require_gt_pose, require_no_missing_frames, require_good_cam_movement, count_max_per_category, sequences_require_mesh, dict_nested_frames_ban: Dict[str, Dict[str, List[str]]]=None):
#         logger.info("filtering frames...")
#         if dict_nested_frames is not None:
#             dict_nested_sequences = {}
#             for category, dict_sequence_frames in dict_nested_frames.items():
#                 if category not in self.categories:
#                     continue
#                 dict_nested_sequences[category] = []
#
#                 if dict_sequence_frames is not None:
#                     for sequence, frames in dict_sequence_frames.items():
#                         dict_nested_sequences[category].append(sequence)
#                 else:
#                     dict_nested_sequences[category] = None
#         else:
#             dict_nested_sequences = None
#
#         dict_nested_sequences_ban = None
#         if dict_nested_frames_ban is not None:
#             for category, dict_sequence_frames in dict_nested_frames_ban.items():
#                 if dict_sequence_frames is not None:
#                     for sequence, frames in dict_sequence_frames.items():
#                         if frames is None:
#                             if dict_nested_sequences_ban is None:
#                                 dict_nested_sequences_ban = {}
#                             if category not in dict_nested_sequences_ban.keys():
#                                 dict_nested_sequences_ban[category] = []
#                             dict_nested_sequences_ban[category].append(sequence)
#                 else:
#                     if dict_nested_sequences_ban is None:
#                         dict_nested_sequences_ban = {}
#                     dict_nested_sequences_ban[category] = None
#
#         # get sequences
#         dict_nested_sequences = CO3D_SequenceMeta.complete_nested_metas(path_meta=self.path_meta, dict_nested_metas=dict_nested_sequences, dict_nested_metas_ban=dict_nested_sequences_ban)
#
#         # filter dict_nested_sequences
#         for i, category in tqdm(enumerate(dict_nested_sequences.keys())):
#             if category not in self.categories:
#                 dict_nested_sequences[category] = []
#                 continue
#             if require_pcl or require_gt_pose or count_max_per_category is not None or sequences_require_mesh or require_good_cam_movement:
#
#                 sequences = [self.get_sequence_by_category_and_name(category=category, name=sequence_name) for sequence_name
#                              in dict_nested_sequences[category]]
#
#                 if require_no_missing_frames:
#                     sequences = list(filter(lambda sequence: sequence.no_missing_frames, sequences))
#
#                 if require_good_cam_movement:
#                     sequences = list(filter(lambda sequence: sequence.good_cam_movement, sequences))
#
#                 #if dict_nested_sequences_ban is not None and category in dict_nested_sequences_ban.keys():
#                 #    sequences = [seq for seq in sequences if seq not in dict_nested_sequences_ban[category]]
#                 if require_gt_pose:
#                     sequences = list(filter(lambda sequence: sequence.gt_pose_available, sequences))
#
#                 if require_pcl:
#                     sequences = list(filter(lambda sequence: sequence.meta.rfpath_pcl != Path('None'), sequences))
#                     if require_pcl_score is not None:
#                         sequences = list(
#                             filter(lambda sequence: sequence.meta.pcl_quality_score > require_pcl_score, sequences))
#                     if sort_pcl_score:
#                         sequences = sorted(sequences, key=lambda sequence: -sequence.meta.pcl_quality_score)
#                 if sequences_require_mesh:
#                     sequences_no_fpath_mesh = list(filter(lambda sequence: not sequence.fpath_mesh.exists(), sequences))
#                     sequences = list(filter(lambda sequence: sequence.fpath_mesh.exists(), sequences))
#                     if len(sequences_no_fpath_mesh) > 0:
#                         sequences_no_fpath_mesh_names = [s.name for s in sequences_no_fpath_mesh]
#                         logger.info(f'Filtering out sequences due to no mesh available for category {category}: \n{sequences_no_fpath_mesh_names}')
#                 if count_max_per_category is not None:
#                     sequences = sequences[:count_max_per_category]
#                 dict_nested_sequences[category] = [sequence.name for sequence in sequences]
#         return dict_nested_sequences
#

#
#
#
#     def preprocess(self, config_preprocess: DictConfig):
#         logger.info("preprocess")
#         for key in config_preprocess.keys():
#             if key == 'label' and config_preprocess.label.get('enabled', False):
#                 override = config_preprocess.label.get('override', False)
#                 self.preprocess_label(override=override)
#             elif key == 'pcl' and config_preprocess.pcl.get('enabled', False):
#                 override = config_preprocess.pcl.get('override', False)
#                 remove_previous = config_preprocess.pcl.get('remove_previous', False)
#                 self.preprocess_pcls(override=override, remove_previous=remove_previous)
#             elif key == 'cuboid' and config_preprocess.cuboid.get('enabled', False):
#                 override = config_preprocess.cuboid.get('override', False)
#                 remove_previous = config_preprocess.cuboid.get('remove_previous', False)
#                 self.preprocess_cuboids(override=override, remove_previous=remove_previous)
#             elif key == 'cuboid_avg' and config_preprocess.cuboid_avg.get('enabled', False):
#                 override = config_preprocess.cuboid_avg.get('override', False)
#                 remove_previous = config_preprocess.cuboid_avg.get('remove_previous', False)
#                 self.preprocess_cuboid_avg(override=override, remove_previous=remove_previous)
#             elif key == 'droid_slam' and config_preprocess.droid_slam.get('enabled', False):
#                 override = config_preprocess.droid_slam.get('override', False)
#                 self.preprocess_droid_slams(override=override)
#             elif key == 'mesh' and config_preprocess.mesh.get('enabled', False):
#                 override = config_preprocess.mesh.get('override', False)
#                 self.preprocess_meshs(override=override)
#             elif key == 'mesh_feats' and config_preprocess.mesh_feats.get('enabled', False):
#                 override = config_preprocess.mesh_feats.get('override', False)
#                 self.preprocess_meshs_feats(override=override)
#
#         # CO3D.preprocess_cam_tform4x4_obj_canonic(config=config)
#         # CO3D.preprocess_front_names(config=config)
#     def preprocess_label(self, override=False, remove_previous=False):
#         logger.info("preprocess labels...")
#         for category, sequences_names in self.dict_category_sequences_names.items():
#             for sequence_name in sequences_names:
#                 logger.info(f"preprocess meshs, sequence {sequence_name}, {self.mesh_name}")
#                 sequence = self.get_sequence_by_category_and_name(category=category, name=sequence_name)
#                 sequence.preprocess_label(override=override)
#
#     def preprocess_meshs(self, override=False, remove_previous=False):
#         logger.info("preprocess meshs...")
#
#         for category, sequences_names in self.dict_category_sequences_names.items():
#             for sequence_name in sequences_names:
#                 logger.info(f"preprocess meshs, sequence {sequence_name}, {self.mesh_name}")
#                 sequence = self.get_sequence_by_category_and_name(category=category, name=sequence_name)
#                 sequence.preprocess_mesh(override=override)
#
#     def preprocess_meshs_feats(self, override=False, remove_previous=False):
#         logger.info("preprocess meshs feats...")
#
#         for category, sequences_names in self.dict_category_sequences_names.items():
#             for sequence_name in sequences_names:
#                 logger.info(f"preprocess meshs feats, sequence {sequence_name}, {self.mesh_name}")
#                 sequence = self.get_sequence_by_category_and_name(category=category, name=sequence_name)
#                 sequence.preprocess_mesh_feats(override=override)
#
#     def preprocess_droid_slams(self, override=False, remove_previous=False):
#         logger.info("preprocess meshs...")
#
#         for category, sequences_names in self.dict_category_sequences_names.items():
#             for sequence_name in sequences_names:
#                 logger.info(f"preprocess droid slam, sequence {sequence_name}")
#                 sequence = self.get_sequence_by_category_and_name(category=category, name=sequence_name)
#                 sequence.preprocess_droid_slam(override=override)
#
#
#     def preprocess_pcls(self, override=False, remove_previous=False):
#         logger.info("preprocess pcls...")
#
#         for category, sequences_names in self.dict_category_sequences_names.items():
#             for sequence_name in sequences_names:
#                 logger.info(f"preprocess pcls, sequence {sequence_name}")
#                 sequence = self.get_sequence_by_category_and_name(category=category, name=sequence_name)
#                 sequence.preprocess_pcl(override=override)
#
#     def preprocess_cuboids(self, override=False, remove_previous=False):
#         logger.info("preprocess cuboids...")
#
#         for category, sequences_names in self.dict_category_sequences_names.items():
#             for sequence_name in sequences_names:
#                 logger.info(f"preprocess cuboids, sequence {sequence_name}")
#                 sequence = self.get_sequence_by_category_and_name(category=category, name=sequence_name)
#                 sequence.preprocess_cuboid(override=override)
#
#     def preprocess_cuboid_avg(self, override=False, remove_previous=False):
#         logger.info("preprocess cuboids avg...")
#         for category in self.categories:
#             fpath = Path(self.path_preprocess).joinpath('cuboids', 'avg', self.name, f'{category}.ply')
#
#             if not fpath.exists() or override:
#                 logger.info(f"preprocessing average cuboid and saving it to {fpath}")
#                 percentile_noise = 0.03
#                 cuboid_pts3d_max_count = 1000
#                 device = 'cuda:0'
#                 from od3d.cv.visual.show import show_pcl
#                 from od3d.cv.geometry.downsample import voxel_downsampling
#                 from od3d.cv.geometry.transform import transf3d_broadcast
#                 from od3d.cv.geometry.primitives import Cuboids
#
#                 cuboids_limits =  []
#                 pcls = []
#                 for sequence_name in self.dict_category_sequences_names[category]:
#
#                     sequence = self.get_sequence_by_category_and_name(category=category, name=sequence_name)
#                     cuboids_limits.append(sequence.cuboid.get_limits())
#
#                     #pcls.append(transf3d_broadcast(voxel_downsampling(sequence.pcl_clean, K=cuboid_pts3d_max_count).to(device=device),
#                     #                               transf4x4=sequence.cuboid_front_tform4x4_obj.to(device=device)))
#                 #pcl_max_pts_id = torch.Tensor([pcl.shape[0] for pcl in pcls]).max(dim=0)[1]
#                 #pcls.append(pcls[0])
#                 #pcls[0] = pcls[pcl_max_pts_id]
#
#                 #cuboid_pts3d = torch.cat(pcls, dim=0)
#                 #cuboids_limits = torch.stack(
#                 #    [cuboid_pts3d.quantile(dim=-2, q=percentile_noise), cuboid_pts3d.quantile(dim=-2, q=1. - percentile_noise)],
#                 #    dim=-2)[None,]
#
#                 cuboids_limits = torch.cat(cuboids_limits, dim=0)
#
#                 #cuboids_limits = torch.stack([cuboids_limits[:, 0].quantile(dim=0, q=percentile_noise), cuboids_limits[:, 1].quantile(dim=0, q=1. - percentile_noise)], dim=0)
#
#                 cuboids_limits = torch.stack([cuboids_limits[:, 0].mean(dim=0), cuboids_limits[:, 1].mean(dim=0)], dim=0)
#
#                 cuboids = Cuboids.create_dense_from_limits(limits=cuboids_limits[None,], verts_count=cuboid_pts3d_max_count)
#
#                 fpath.parent.mkdir(parents=True, exist_ok=True)
#                 save_ply(fpath, verts=cuboids.verts, faces=cuboids.faces)
#                 # show_pcl([cuboids.verts.to(device=device)] + pcls)
#
