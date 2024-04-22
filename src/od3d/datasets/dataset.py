import logging
logger = logging.getLogger(__name__)
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig
from enum import Enum
from pathlib import Path
import torch
from typing import List, Dict
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame, OD3D_FrameMeta
from od3d.datasets.sequence_meta import OD3D_SequenceMetaCategoryMixin
from od3d.datasets.frames import OD3D_Frames
from od3d.data import ExtEnum
import inspect
from tqdm import tqdm
import numpy as np
import od3d.io
from od3d.datasets.frame import OD3D_FRAME_MASK_TYPES, OD3D_FRAME_DEPTH_TYPES
from od3d.cv.geometry.transform import proj3d2d_broadcast
from od3d.datasets.sequence import OD3D_Sequence
from od3d.datasets.sequence_meta import OD3D_SequenceMeta

class OD3D_SEQ_MODALITIES(str, Enum):
    PCL = 'pcl'

class OD3D_PREPROCESS_MODALITIES(str, Enum):
    FRONT_FRAME = 'front_frame'
    CUBOID = 'cuboid'
    CUBOID_AVG = 'cuboid_avg'
    PCL = 'pcl'
    MASK = 'mask'

class OD3D_DATASET_SPLITS(str, ExtEnum):
    SEQUENCES_SEPARATED = 'sequences_separated'
    RANDOM = 'random'
    SEQUENCES_SHARED = 'sequences_shared'

class OD3D_Dataset(Dataset):
    from od3d.datasets.enum import OD3D_CATEGORIES
    map_od3d_categories = None
    all_categories = list(OD3D_CATEGORIES)
    subclasses = {}
    frame_type = OD3D_Frame
    modalities = None
    path_raw: Path = None
    path_preprocess: Path = None
    categories: List[str] = None
    transform = None
    index_shift = 0
    subset_fraction = 1.
    dict_nested_frames: Dict = None
    dict_nested_frames_ban: Dict = None

    @classmethod
    def create_from_config(cls, config: DictConfig, transform=None):
        if config.get("setup", False).get("enabled", False):
            cls.setup(config=config)
        if config.get("extract_meta", False).get("enabled", False):
            cls.extract_meta(config=config)

        keys = inspect.getfullargspec(cls.__init__)[0][1:]
        od3d_dataset = cls(**dict((key, config.get(key)) for key in keys if config.get(key, None) is not None), transform=transform)

        if config.get("preprocess", False):
            od3d_dataset.preprocess(config_preprocess=config.preprocess)

        return od3d_dataset

    def get_as_dict(self):
        from od3d.cv.transforms.transform import OD3D_Transform
        _dict = {}
        keys = inspect.getfullargspec(self.__init__)[0][1:]
        for key in keys:
            if hasattr(self, key):
                _dict[key] = getattr(self, key)
                if isinstance(_dict[key], Enum):
                    _dict[key] = str(_dict[key])
                if isinstance(_dict[key], OD3D_Transform):
                    _dict[key] = _dict[key].get_as_dict()
        _dict['class_name'] = type(self).__name__
        return _dict

    def save_to_config(self, fpath: Path):
        _dict = self.get_as_dict()
        from od3d.io import write_dict_as_yaml
        write_dict_as_yaml(fpath=fpath, _dict=_dict, save_enum_as_str=True)

    @classmethod
    def create_by_name(cls, name: str, config: dict = None):
        config_loaded = od3d.io.read_config_intern(rfpath=Path("datasets").joinpath(f"{name}.yaml"))
        if config is not None:
            OmegaConf.set_struct(config_loaded, False)
            config_loaded = OmegaConf.merge(config_loaded, config)
            OmegaConf.set_struct(config_loaded, True)
        return cls.create_from_config(config_loaded)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[str]=None, transform=None, index_shift=0, subset_fraction=1.,
                 dict_nested_frames: Dict=None, dict_nested_frames_ban: Dict=None):



        logger.info(f'init dataset {name}...')


        if categories is not None:
            if self.map_od3d_categories is not None:
                self.categories = [self.map_od3d_categories.get(category, category) if category not in self.all_categories else category for category in categories]
            else:
                self.categories = categories
        else:
            self.categories = self.all_categories

        self.name = name
        self.path_raw: Path = Path(path_raw)
        self.path_preprocess: Path = Path(path_preprocess)
        self.subset_fraction: float = subset_fraction

        if transform is None:
            from od3d.cv.transforms.rgb_uint8_to_float import RGB_UInt8ToFloat
            transform = RGB_UInt8ToFloat()

        self.transform = transform
        self.index_shift = index_shift
        self.modalities = modalities
        self.splits_featured = [OD3D_DATASET_SPLITS.RANDOM]

        logger.info('completing nested frames..., can take up to 500 seconds...')
        dict_nested_frames = self.frame_type.meta_type.complete_nested_metas(path_meta=self.path_meta,
                                                                             dict_nested_metas=dict_nested_frames,
                                                                             dict_nested_metas_ban=
                                                                             dict_nested_frames_ban)


        dict_nested_frames = self.filter_dict_nested_frames(dict_nested_frames)

        logger.info('unrolling nested frames...')
        list_frames_unique = self.frame_type.meta_type.unroll_nested_metas(dict_nested_meta=dict_nested_frames)

        logger.info('filtering frames...')
        list_frames_unique = self.filter_list_frames_unique(list_frames_unique)

        self.frames_count = len(list_frames_unique)
        if self.subset_fraction is not None and self.subset_fraction != 1.:
            logger.info('filtering with subset fraction...')
            frames_ids_subset = self.get_subset_item_ids(subset_fraction=subset_fraction)
            list_frames_unique = [list_frames_unique[id] for id in frames_ids_subset]

        self.set_list_frames_unique(list_frames_unique=list_frames_unique)
        logger.info(f"found {self.frames_count} frames.")

    def filter_dict_nested_frames(self, dict_nested_frames):
        return dict_nested_frames

    def filter_list_frames_unique(self, list_frames_unique):
        return list_frames_unique

    def __len__(self):
        return self.frames_count

    def get_subset_with_dict_nested_frames(self, dict_nested_frames: Dict):
        import copy
        dataset = copy.deepcopy(self)
        dict_nested_frames_compl = OD3D_FrameMeta.complete_nested_metas(path_meta=self.path_meta, dict_nested_metas=dict_nested_frames)
        list_frames_unique = OD3D_FrameMeta.unroll_nested_metas(dict_nested_meta=dict_nested_frames_compl)
        dataset.set_list_frames_unique(list_frames_unique=list_frames_unique)
        return dataset

    def set_list_frames_unique(self, list_frames_unique):
        self.list_frames_unique = list_frames_unique
        self.dict_nested_frames = OD3D_FrameMeta.rollup_flattened_frames(
            list_meta_names_unique=self.list_frames_unique)
        self.frames_count = len(self.list_frames_unique)

    def get_subset_with_item_ids(self, item_ids):
        list_frames_unique = [self.list_frames_unique[id] for id in item_ids]
        dict_nested_frames = OD3D_FrameMeta.rollup_flattened_frames(list_meta_names_unique=list_frames_unique)
        return self.get_subset_with_dict_nested_frames(dict_nested_frames)

    def get_subset_with_names_unique(self, list_frames_unique: List[str]):
        dict_nested_frames = OD3D_FrameMeta.rollup_flattened_frames(list_meta_names_unique=list_frames_unique)
        return self.get_subset_with_dict_nested_frames(dict_nested_frames)

    def get_subset_item_ids(self, subset_fraction):
        return torch.multinomial(torch.ones(size=(len(self),)),
                                 num_samples=int(subset_fraction * len(self)),
                                 replacement=False)

    def get_split_item_ids(self, fraction1: float):
        item_ids_subsetA = self.get_subset_item_ids(subset_fraction=fraction1)
        item_ids = torch.arange(len(self))
        item_ids_maskA = torch.zeros(size=(len(self),), dtype=torch.bool, device=item_ids_subsetA.device)
        item_ids_maskA[item_ids_subsetA] = True
        item_ids_subsetA = item_ids[item_ids_maskA]
        item_ids_subsetB = item_ids[~item_ids_maskA]

        return item_ids_subsetA, item_ids_subsetB
    def get_fractionA_from_fractionA_and_fraction_B(self, fraction1: float, fraction2: float=None):
        if fraction2 is None:
            assert fraction1 > 0. and fraction1 < 1.
        else:
            fraction12 = fraction1 + fraction2
            if fraction12 != 1.:
                fraction1 = fraction1 / fraction12
                fraction2 = fraction2 / fraction12
                logger.warning(f'Subset fractions dont sum up to 1. Setting 1.={fraction1}, 2.={fraction2}')
        return fraction1

    def get_split(self, fraction1: float, fraction2: float, split: OD3D_DATASET_SPLITS = OD3D_DATASET_SPLITS.RANDOM):
        fraction1 = self.get_fractionA_from_fractionA_and_fraction_B(fraction1, fraction2)
        if split == OD3D_DATASET_SPLITS.SEQUENCES_SHARED:
            return self.get_split_sequences_shared(fraction1=fraction1)
        elif split == OD3D_DATASET_SPLITS.RANDOM:
            return self.get_split_random(fraction1=fraction1)
        elif split == OD3D_DATASET_SPLITS.SEQUENCES_SEPARATED:
            return self.get_split_sequences_separated(fraction1=fraction1)
        else:
            logger.warning(f'Unknown split {split}.')


    def get_split_random(self, fraction1: float):
        item_ids_subsetA, item_ids_subsetB = self.get_split_item_ids(fraction1=fraction1)
        co3d_subsetA = self.get_subset_with_item_ids(item_ids=item_ids_subsetA)
        co3d_subsetB = self.get_subset_with_item_ids(item_ids=item_ids_subsetB)
        return co3d_subsetA, co3d_subsetB

    def get_split_sequences_shared(self, fraction1: float):
        raise NotImplementedError

    def get_split_sequences_separated(self, fraction1: float):
        raise NotImplementedError

    def __getitem__(self, item):
        item_id_shift = (item + self.index_shift) % len(self)
        frame = self.transform(self.get_item(item_id_shift))
        frame.item_id = item_id_shift
        return frame

    def get_item(self, item):
        return self.get_frame_by_name_unique(name_unique=self.list_frames_unique[item])

    def get_frame_by_name_unique(self, name_unique: str):
        raise NotImplementedError

    def get_random_item(self):
        return self.__getitem__(np.random.choice(self.__len__()))

    def get_item_id_by_name_unique(self, name_unique: str):
        for i in range(len(self)):
            if self.list_frames_unique[i] == name_unique:
                return i
        return -1


    def collate_fn(self, frames: List[OD3D_Frame], device='cpu', dtype=torch.float32, modalities=None):
        if modalities is None:
            modalities = self.modalities
        frames = OD3D_Frames.get_frames_from_list(frames, modalities=modalities, dtype=dtype, device=device)
        return frames

    def get_dataloader(self, batch_size=1, shuffle=False):
        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        return dataloader

    @staticmethod
    def setup(config: DictConfig):
        raise NotImplementedError

    @staticmethod
    def extract_meta(config: DictConfig):
        raise NotImplementedError

    @staticmethod
    def record_video(config: DictConfig):
        raise NotImplementedError

    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            if key == 'mask' and config_preprocess.mask.get('enabled', False):
                override = config_preprocess.mask.get('override', False)
                self.preprocess_mask(override=override)



    def preprocess_mask(self, override=False, remove_previous=False):
        logger.info("preprocess masks...")
        from functools import partial

        first_frame = self.get_frame_by_name_unique(self.list_frames_unique[0])
        first_frame_fpath_mask = first_frame.fpath_mask
        mask_type = first_frame.mask_type
        if first_frame_fpath_mask.exists() and not override:
            logger.info(f"masks exists, at least at {first_frame_fpath_mask}, skip preprocess mask")
            return

        modalities = [OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.CAM_INTR4X4,
                      OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ, OD3D_FRAME_MODALITIES.SIZE]
        if mask_type == OD3D_FRAME_MASK_TYPES.SAM_SFM_RAYS_CENTER3D:
            modalities.append(OD3D_FRAME_MODALITIES.RAYS_CENTER3D)
        elif mask_type == OD3D_FRAME_MASK_TYPES.MESH:
            modalities.append(OD3D_FRAME_MODALITIES.MESH)
        elif mask_type == OD3D_FRAME_MASK_TYPES.SAM:
            pass
        else:
            raise NotImplementedError

        modalities_orig = self.modalities
        self.modalities = modalities
        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=1, shuffle=False,
                                                 collate_fn=partial(self.collate_fn, modalities=modalities))
        logging.info(f"Dataset contains {len(self)} frames.")

        if mask_type == OD3D_FRAME_MASK_TYPES.SAM_SFM_RAYS_CENTER3D or mask_type == OD3D_FRAME_MASK_TYPES.SAM:
            from od3d.models.model import OD3D_Model
            model = OD3D_Model.create_by_name('sam')
            model.cuda()
            model.eval()
            self.transform = model.transform
        else:
            pass

        from od3d.cv.io import get_default_device
        device = get_default_device()
        for batch in iter(dataloader):
            logger.info(f'{batch.name_unique[0]}')  # sequence_name[0]}')
            frames = [self.get_frame_by_name_unique(name_unique=name_unique) for name_unique in batch.name_unique]
            batch.to(device=device)

            if mask_type == OD3D_FRAME_MASK_TYPES.MESH:
                masks = batch.mesh.render_feats(cams_tform4x4_obj=batch.cam_tform4x4_obj.to(device=device),
                                                cams_intr4x4=batch.cam_intr4x4.to(device=device),
                                                imgs_sizes=batch.size.to(device=device), modality='mask')

                for b in range(len(batch.name_unique)):
                    frame = frames[b]
                    mask = masks[b]
                    frame.write_mask(mask)
            else:
                if mask_type == OD3D_FRAME_MASK_TYPES.SAM_SFM_RAYS_CENTER3D:
                    center_pxl2d = proj3d2d_broadcast(proj4x4=batch.cam_proj4x4_obj, pts3d=batch.rays_center3d)
                elif mask_type == OD3D_FRAME_MASK_TYPES.SAM:
                    center_pxl2d = batch.size[None, [1,0]] / 2
                else:
                    raise ValueError(f"mask_type {mask_type} not supported")
                masks, scores, logits = model(batch.rgb, center_pxl2d)

                for b in range(len(batch.name_unique)):
                    masks_b = masks[b]
                    frame = frames[b]
                    sam_lvl = scores[b].argmax()
                    mask = masks[b, sam_lvl:sam_lvl+1]
                    # from od3d.cv.visual.draw import draw_pixels
                    # mask = draw_pixels(mask, pxls=center_pxl2d[b:b+1])
                    frame.write_mask(mask)
        self.modalities = modalities_orig


    def preprocess_depth(self, override=False, remove_previous=False):
        logger.info("preprocess depth...")
        from functools import partial

        first_frame = self.get_frame_by_name_unique(self.list_frames_unique[0])
        first_frame_fpath_depth = first_frame.fpath_depth
        depth_type = first_frame.depth_type
        if first_frame_fpath_depth.exists() and not override:
            logger.info(f"masks exists, at least at {first_frame_fpath_depth}, skip preprocess mask")
            return

        modalities = [OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.CAM_INTR4X4,
                      OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ, OD3D_FRAME_MODALITIES.SIZE]
        if depth_type == OD3D_FRAME_DEPTH_TYPES.MESH:
            modalities.append(OD3D_FRAME_MODALITIES.MESH)
        else:
            raise NotImplementedError

        modalities_orig = self.modalities
        self.modalities = modalities
        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=1, shuffle=False,
                                                 collate_fn=partial(self.collate_fn, modalities=modalities))
        logging.info(f"Dataset contains {len(self)} frames.")

        from od3d.cv.io import get_default_device
        device = get_default_device()
        for batch in iter(dataloader):
            logger.info(f'{batch.name_unique[0]}')  # sequence_name[0]}')
            frames = [self.get_frame_by_name_unique(name_unique=name_unique) for name_unique in batch.name_unique]
            batch.to(device=device)

            if depth_type == OD3D_FRAME_MASK_TYPES.MESH:
                depths = batch.mesh.render_feats(cams_tform4x4_obj=batch.cam_tform4x4_obj.to(device=device),
                                                 cams_intr4x4=batch.cam_intr4x4.to(device=device),
                                                 imgs_sizes=batch.size.to(device=device), modality='depth')

                depths_masks = batch.mesh.render_feats(cams_tform4x4_obj=batch.cam_tform4x4_obj.to(device=device),
                                                 cams_intr4x4=batch.cam_intr4x4.to(device=device),
                                                 imgs_sizes=batch.size.to(device=device), modality='mask')

                for b in range(len(batch.name_unique)):
                    frame = frames[b]
                    depth = depths[b]
                    depth_mask = depths_masks[b]
                    frame.write_depth(depth)
                    frame.write_depth_mask(depth_mask)

    def visualize(self, item: int):
        raise NotImplementedError

    @property
    def path_meta(self):
        return self.path_preprocess.joinpath('meta')

    @staticmethod
    def get_path_meta(config):
        return OD3D_Dataset.get_path_meta_from_path_preprocess(OD3D_Dataset.get_path_preprocess(config=config))

    @staticmethod
    def get_path_meta_from_path_preprocess(path_preprocess: Path):
        return path_preprocess.joinpath('meta')

    @staticmethod
    def get_path_preprocess(config):
        return Path(config.path_preprocess)

    @staticmethod
    def get_path_raw(config):
        return Path(config.path_raw)

    def get_frames_categories(self, max_frames_count_per_category=1, filter_single_category=False):
        dict_frames = {category: [] for category in self.categories}
        categories_filled = []
        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=10, shuffle=True,
                                                 collate_fn=self.collate_fn)
        for i, batch in enumerate(tqdm(dataloader)):
            for b in range(len(batch)):
                if batch.category is not None:
                    categories = [batch.category[b]]
                elif batch.categories is not None and (not filter_single_category or len(batch.categories[b]) == 1):
                    categories = list(set(batch.categories[b]))
                else:
                    categories = []
                for category in categories:
                    if category not in self.categories:
                        continue
                    # logger.info(batch.categories[b])
                    if len(dict_frames[category]) < max_frames_count_per_category:
                        dict_frames[category].append(batch.rgb[b])
                    else:
                        categories_filled.append(category)
                        categories_filled = list(set(categories_filled))
            if len(categories_filled) == len(self.categories):
                break

        dict_frames_stacked = {}
        for category in dict_frames.keys():
            if len(dict_frames[category]) > 0:
                dict_frames_stacked[category] = torch.stack(list(dict_frames[category]), dim=0)

        return dict_frames_stacked



class OD3D_SequenceDataset(OD3D_Dataset):
    sequence_type = OD3D_Sequence
    frames_count_max_per_sequence = None
    sequences_count_max_per_category = None

    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES],
                 path_raw: Path, path_preprocess: Path,
                 categories: List=None,
                 dict_nested_frames: Dict=None,
                 dict_nested_frames_ban: Dict=None,
                 transform=None, index_shift=0, subset_fraction=1., frames_count_max_per_sequence=None,
                 sequences_count_max_per_category=None):

        self.frames_count_max_per_sequence = frames_count_max_per_sequence
        self.sequences_count_max_per_category = sequences_count_max_per_category
        if categories is not None:
            if self.map_od3d_categories is not None:
                self.categories = [self.map_od3d_categories.get(category, category) if category not in self.all_categories else category for category in categories]
            else:
                self.categories = categories
        else:
            self.categories = self.all_categories
        self.path_raw = Path(path_raw)
        self.path_preprocess = Path(path_preprocess)
        self.modalities = modalities

        logger.info("filtering sequences...")
        self.dict_category_sequences_names = self.filter_dict_nested_sequences(dict_nested_frames=dict_nested_frames,
                                                                               dict_nested_frames_ban=dict_nested_frames_ban,
                                                                               count_max_per_category=self.sequences_count_max_per_category)

        logger.info(f'sequences filtered')
        sequences_filtered_str = '\n'
        for category in self.dict_category_sequences_names.keys():
            if len(self.dict_category_sequences_names[category]) > 0:
                sequences_filtered_str += category + ': \n'
                for sequence_name in self.dict_category_sequences_names[category]:
                    sequences_filtered_str += f"  '{sequence_name}':\n"
        logger.info(sequences_filtered_str)

        logger.info(f'Found sequences per category')

        dict_nested_frames_seqs_filtered = {}
        for category in self.dict_category_sequences_names.keys(): #.keys():
            logger.info(f'{category}: {len(self.dict_category_sequences_names[category])}')
            if len(self.dict_category_sequences_names[category]) == 0:
                continue
            if dict_nested_frames is not None and category in dict_nested_frames.keys():
                dict_nested_frames_seqs_filtered[category] = dict_nested_frames[category]
            else:
                if dict_nested_frames is None:
                    dict_nested_frames_seqs_filtered[category] = None
                else:
                    # category not in dict_nested_frames
                    dict_nested_frames_seqs_filtered[category] = {}

            for sequence_name in self.dict_category_sequences_names[category]:
                if dict_nested_frames is not None and category in dict_nested_frames.keys() and dict_nested_frames[category] is not None and sequence_name in dict_nested_frames[category]:
                    dict_nested_frames_seqs_filtered[category][sequence_name] = dict_nested_frames[category][sequence_name]
                else:
                    if dict_nested_frames is None or (category in dict_nested_frames and dict_nested_frames_seqs_filtered[category] is None):
                        if not isinstance(dict_nested_frames_seqs_filtered[category], dict):
                            dict_nested_frames_seqs_filtered[category] = {}
                        dict_nested_frames_seqs_filtered[category][sequence_name] = None
                    else:
                        # category / sequence not in dict_nested_frames
                        dict_nested_frames_seqs_filtered[category][sequence_name] = []
        dict_nested_frames = dict_nested_frames_seqs_filtered
        super().__init__(categories=categories, dict_nested_frames=dict_nested_frames,
                         dict_nested_frames_ban=dict_nested_frames_ban, name=name, modalities=modalities,
                         path_raw=path_raw, path_preprocess=path_preprocess, transform=transform, index_shift=index_shift,
                         subset_fraction=subset_fraction)

    def filter_dict_nested_frames(self, dict_nested_frames: Dict[str, Dict[str, List[str]]]):
        dict_nested_frames = super().filter_dict_nested_frames(dict_nested_frames=dict_nested_frames)

        frames_count_max_per_sequence = self.frames_count_max_per_sequence

        # filter frames to exist in sequences:
        dict_nested_frames_filtered = {}
        for category, sequences in self.dict_category_sequences_names.items():
            dict_nested_frames_filtered[category] = {}
            for sequence in sequences:
                dict_nested_frames_filtered[category][sequence] = dict_nested_frames[category][sequence]
        dict_nested_frames = dict_nested_frames_filtered

        if frames_count_max_per_sequence is not None:
            dict_nested_frames = self.frame_type.meta_type.complete_nested_metas(path_meta=self.path_meta,
                                                                                 dict_nested_metas=dict_nested_frames)
            dict_nested_frames_filtered = {}
            for category, dict_sequence_name_frames_names in dict_nested_frames.items():
                for sequence_name, frames_names in dict_sequence_name_frames_names.items():
                    frames_names_filtered = self.sequence_type.get_subset_frames_names_uniform(frames_names, count_max_per_sequence=frames_count_max_per_sequence)
                    if category not in dict_nested_frames_filtered.keys():
                        dict_nested_frames_filtered[category] = {}
                    dict_nested_frames_filtered[category][sequence_name] = frames_names_filtered
            dict_nested_frames = dict_nested_frames_filtered

        return dict_nested_frames


    def filter_dict_nested_sequences(self, dict_nested_frames: Dict[str, Dict[str, List[str]]], dict_nested_frames_ban: Dict[str, Dict[str, List[str]]]=None, count_max_per_category=None):
        logger.info("filtering frames...")
        if dict_nested_frames is not None:
            dict_nested_sequences = {}
            for category, dict_sequence_frames in dict_nested_frames.items():
                if category not in self.categories:
                    continue
                dict_nested_sequences[category] = []

                if dict_sequence_frames is not None:
                    for sequence, frames in dict_sequence_frames.items():
                        dict_nested_sequences[category].append(sequence)
                else:
                    dict_nested_sequences[category] = None
        else:
            dict_nested_sequences = None

        dict_nested_sequences_ban = None
        if dict_nested_frames_ban is not None:
            for category, dict_sequence_frames in dict_nested_frames_ban.items():
                if dict_sequence_frames is not None:
                    for sequence, frames in dict_sequence_frames.items():
                        if frames is None:
                            if dict_nested_sequences_ban is None:
                                dict_nested_sequences_ban = {}
                            if category not in dict_nested_sequences_ban.keys():
                                dict_nested_sequences_ban[category] = []
                            dict_nested_sequences_ban[category].append(sequence)
                else:
                    if dict_nested_sequences_ban is None:
                        dict_nested_sequences_ban = {}
                    dict_nested_sequences_ban[category] = None

        # get sequences
        dict_nested_sequences = OD3D_SequenceMetaCategoryMixin.complete_nested_metas(path_meta=self.path_meta, dict_nested_metas=dict_nested_sequences, dict_nested_metas_ban=dict_nested_sequences_ban)

        # filter dict_nested_sequences
        for i, category in tqdm(enumerate(dict_nested_sequences.keys())):
            if category not in self.categories:
                dict_nested_sequences[category] = []
                continue
            if count_max_per_category is not None:
                dict_nested_sequences[category] = dict_nested_sequences[category][:count_max_per_category]

            #
            # if require_pcl or require_gt_pose or count_max_per_category is not None or sequences_require_mesh or require_good_cam_movement:
            #
            #     sequences = [self.get_sequence_by_category_and_name(category=category, name=sequence_name) for sequence_name
            #                  in dict_nested_sequences[category]]
            #
            #     if require_no_missing_frames:
            #         sequences = list(filter(lambda sequence: sequence.no_missing_frames, sequences))
            #
            #     if require_good_cam_movement:
            #         sequences = list(filter(lambda sequence: sequence.good_cam_movement, sequences))
            #
            #     #if dict_nested_sequences_ban is not None and category in dict_nested_sequences_ban.keys():
            #     #    sequences = [seq for seq in sequences if seq not in dict_nested_sequences_ban[category]]
            #     if require_gt_pose:
            #         sequences = list(filter(lambda sequence: sequence.gt_pose_available, sequences))
            #
            #     if require_pcl:
            #         sequences = list(filter(lambda sequence: sequence.meta.rfpath_pcl != Path('None'), sequences))
            #         if require_pcl_score is not None:
            #             sequences = list(
            #                 filter(lambda sequence: sequence.meta.pcl_quality_score > require_pcl_score, sequences))
            #         if sort_pcl_score:
            #             sequences = sorted(sequences, key=lambda sequence: -sequence.meta.pcl_quality_score)
            #     if sequences_require_mesh:
            #         sequences_no_fpath_mesh = list(filter(lambda sequence: not sequence.fpath_mesh.exists(), sequences))
            #         sequences = list(filter(lambda sequence: sequence.fpath_mesh.exists(), sequences))
            #         if len(sequences_no_fpath_mesh) > 0:
            #             sequences_no_fpath_mesh_names = [s.name for s in sequences_no_fpath_mesh]
            #             logger.info(f'Filtering out sequences due to no mesh available for category {category}: \n{sequences_no_fpath_mesh_names}')
            #             if count_max_per_category is not None:
            #                 sequences = sequences[:count_max_per_category]
            #             dict_nested_sequences[category] = [sequence.name for sequence in sequences]
        return dict_nested_sequences


    def get_subset_by_sequences(self, dict_category_sequences: Dict[str, List[str]], frames_count_max_per_sequence=None):
        dict_nested_frames = {}
        for cat, seqs_names in dict_category_sequences.items():
            dict_nested_frames[cat] = {}
            for seq_name in seqs_names:
                seq = self.get_sequence_by_name_unique(name_unique=f'{cat}/{seq_name}')
                dict_nested_frames[cat][seq_name] = OD3D_Sequence.get_subset_frames_names_uniform(
                    frames_names=seq.frames_names, count_max_per_sequence=frames_count_max_per_sequence)

        return self.get_subset_with_dict_nested_frames(dict_nested_frames)
        #return OD3D_SequenceDataset(
        #    name=self.name, modalities=self.modalities, path_raw=self.path_raw, path_preprocess=self.path_preprocess,
        #    categories=self.categories, dict_nested_frames=dict_nested_frames, transform=self.transform,
        #    index_shift=self.index_shift)

    def get_split_sequences_shared(self, fraction1: float):
        dict_category_sequence_name_frames_names_subsetA = {}
        dict_category_sequence_name_frames_names_subsetB = {}
        dict_category_sequence_name_frames_names = self.frame_type.meta_type.rollup_flattened_frames(self.list_frames_unique)
        #dict_category_sequence_name_frames_names = self.list_categories_sequences_names_frames_names_to_dict(self.list_frames_unique)
        for category, dict_sequence_name_frames_names in dict_category_sequence_name_frames_names.items():
            dict_category_sequence_name_frames_names_subsetA[category] = {}
            dict_category_sequence_name_frames_names_subsetB[category] = {}
            for sequence_name, frames_names in dict_sequence_name_frames_names.items():
                frames_names = sorted(frames_names, key=lambda fn: int(fn))
                cutoff = int(len(frames_names) * fraction1)
                dict_category_sequence_name_frames_names_subsetA[category][sequence_name] = frames_names[:cutoff]
                dict_category_sequence_name_frames_names_subsetB[category][sequence_name] = frames_names[cutoff:]

        return self.get_split_from_dicts(dict_category_sequence_name_frames_names_subsetA, dict_category_sequence_name_frames_names_subsetB)

    def get_split_sequences_separated(self, fraction1: float):
        dict_category_sequence_name_frames_names_subsetA = {}
        dict_category_sequence_name_frames_names_subsetB = {}

        dict_category_sequence_name_frames_names = self.frame_type.meta_type.rollup_flattened_frames(self.list_frames_unique)
        #dict_category_sequence_name_frames_names = self.list_categories_sequences_names_frames_names_to_dict(self.list_frames_unique)
        for category, dict_sequence_name_frames_names in dict_category_sequence_name_frames_names.items():
            seqs_names = list(dict_sequence_name_frames_names.keys())
            cutoff = int(len(seqs_names) * fraction1)
            dict_category_sequence_name_frames_names_subsetA[category] = {s: dict_sequence_name_frames_names[s] for s in seqs_names[:cutoff]}
            dict_category_sequence_name_frames_names_subsetB[category] = {s: dict_sequence_name_frames_names[s] for s in seqs_names[cutoff:]}

        return self.get_split_from_dicts(dict_category_sequence_name_frames_names_subsetA, dict_category_sequence_name_frames_names_subsetB)

    #def get_subset_with_dict_nested_frames(self, dict_nested_frames):
    #    return OD3D_SequenceDataset(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
    #                path_preprocess=self.path_preprocess, categories=self.categories,
    #                dict_nested_frames=dict_nested_frames, transform=self.transform, index_shift=self.index_shift)

    def get_split_from_dicts(self, dict_nested_frames_subsetA, dict_nested_frames_subsetB):
        co3d_subsetA = self.get_subset_with_dict_nested_frames(dict_nested_frames_subsetA)
        co3d_subsetB = self.get_subset_with_dict_nested_frames(dict_nested_frames_subsetB)

        return co3d_subsetA, co3d_subsetB

    def preprocess_sfm(self, override=False):
        logger.info("preprocess sfm...")
        from od3d.datasets.sequence_meta import OD3D_SequenceMeta
        for sequence_name_unique in OD3D_SequenceMeta.unroll_nested_metas(self.dict_category_sequences_names):
            sequence = self.get_sequence_by_name_unique(name_unique=sequence_name_unique)
            sequence.preprocess_sfm(override=override)

    def preprocess_pcl(self, override=False):
        logger.info("preprocess pcl...")
        from od3d.datasets.sequence_meta import OD3D_SequenceMeta
        for sequence_name_unique in OD3D_SequenceMeta.unroll_nested_metas(self.dict_category_sequences_names):
            sequence = self.get_sequence_by_name_unique(name_unique=sequence_name_unique)
            sequence.preprocess_pcl(override=override)

    def preprocess_mesh(self, override=False):
        logger.info("preprocess mesh...")
        from od3d.datasets.sequence_meta import OD3D_SequenceMeta
        for sequence_name_unique in OD3D_SequenceMeta.unroll_nested_metas(self.dict_category_sequences_names):
            sequence = self.get_sequence_by_name_unique(name_unique=sequence_name_unique)
            sequence.preprocess_mesh(override=override)

    def preprocess_mesh_feats(self, override=False):
        logger.info("preprocess mesh feats...")
        from od3d.datasets.sequence_meta import OD3D_SequenceMeta
        for sequence_name_unique in OD3D_SequenceMeta.unroll_nested_metas(self.dict_category_sequences_names):
            sequence = self.get_sequence_by_name_unique(name_unique=sequence_name_unique)
            sequence.preprocess_mesh_feats(override=override)

    def preprocess_mesh_feats_dist(self, override=False):
        logger.info("preprocess mesh feats dist...")
        from od3d.datasets.sequence_meta import OD3D_SequenceMeta
        sequences_names_unique = OD3D_SequenceMeta.unroll_nested_metas(self.dict_category_sequences_names)
        for sequence_name_unique1 in sequences_names_unique:
            for sequence_name_unique2 in sequences_names_unique:
                sequence1 = self.get_sequence_by_name_unique(name_unique=sequence_name_unique1)
                sequence2 = self.get_sequence_by_name_unique(name_unique=sequence_name_unique2)
                sequence1.preprocess_mesh_feats_dist(sequence=sequence2, override=override)

    def preprocess_tform_obj(self, override=False):
        logger.info("preprocess tform obj...")
        for sequence_name_unique in OD3D_SequenceMeta.unroll_nested_metas(self.dict_category_sequences_names):
            sequence = self.get_sequence_by_name_unique(name_unique=sequence_name_unique)
            sequence.preprocess_tform_obj(override=override)

    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            if key == 'sfm' and config_preprocess.sfm.get('enabled', False):
                override = config_preprocess.sfm.get('override', False)
                self.preprocess_sfm(override=override)
            if key == 'mask' and config_preprocess.mask.get('enabled', False):
                override = config_preprocess.mask.get('override', False)
                self.preprocess_mask(override=override)
            if key == 'pcl' and config_preprocess.pcl.get('enabled', False):
                override = config_preprocess.pcl.get('override', False)
                self.preprocess_pcl(override=override)
            if key == 'tform_obj' and config_preprocess.tform_obj.get('enabled', False):
                override = config_preprocess.tform_obj.get('override', False)
                self.preprocess_tform_obj(override=override)
            if key == 'mesh' and config_preprocess.mesh.get('enabled', False):
                override = config_preprocess.mesh.get('override', False)
                self.preprocess_mesh(override=override)
            if key == 'mesh_feats' and config_preprocess.mesh_feats.get('enabled', False):
                override = config_preprocess.mesh_feats.get('override', False)
                self.preprocess_mesh_feats(override=override)
            if key == 'mesh_feats_dist' and config_preprocess.mesh_feats_dist.get('enabled', False):
                override = config_preprocess.mesh_feats_dist.get('override', False)
                self.preprocess_mesh_feats_dist(override=override)
    def get_sequence_by_name_unique(self, name_unique: str):
        raise NotImplementedError

    def get_sequences(self):
        sequences = []
        for sequence_name_unique in OD3D_SequenceMeta.unroll_nested_metas(self.dict_category_sequences_names):
            sequences.append(self.get_sequence_by_name_unique(name_unique=sequence_name_unique))
        return sequences

    def save_sequences_as_video(self, H=1080, W=1920, fps=15, fpath_video=None, imgs_count=150):

        if fpath_video is None:
            fpath_video = Path(f'{self.name}.avi')

        sequences = self.get_sequences()
        category_sequences_count = {}
        category_sequences_count_max = 0

        for category in tqdm(self.categories):
            category_sequences_count[category] = sum([1 for sequence in sequences if sequence.category == category])
            if category_sequences_count_max < category_sequences_count[category]:
                category_sequences_count_max = category_sequences_count[category]
            category_sequences_count[category] = 0

        H_cell = H // len(self.categories)
        W_cell = W // category_sequences_count_max
        tstamp_category_sequence_imgs = torch.zeros(size=(imgs_count, len(self.categories), category_sequences_count_max, 3, H_cell, W_cell))

        from od3d.cv.visual.resize import resize
        for sequence in tqdm(sequences):
            cams_tform4x4_world, cams_intr4x4, cams_imgs = sequence.read_cams(cams_count=imgs_count)
            cams_imgs = resize(torch.stack(cams_imgs, dim=0), H_out=H_cell, W_out=W_cell)[:imgs_count]
            tstamp_category_sequence_imgs[:len(cams_imgs), sequence.category_id, category_sequences_count[sequence.category]] = cams_imgs
            category_sequences_count[sequence.category] += 1

        from od3d.cv.visual.show import imgs_to_img
        tstamp_category_sequence_imgs = [imgs_to_img(imgs, H_out=H, W_out=W) for imgs in tstamp_category_sequence_imgs]

        from od3d.cv.visual.video import save_video
        save_video(fpath=fpath_video, imgs=tstamp_category_sequence_imgs, fps=fps)

    def visualize_category_frames(self, imgs_count=5, viewpoints_count=16, H=1080, W=1980, ref_count=1):
        sequences = self.get_sequences()
        from od3d.cv.visual.resize import resize
        from od3d.cv.visual.show import show_scene, show_imgs

        for category in tqdm(self.categories):
            category_mesh = None
            category_cams_imgs = []
            category_cams_tform4x4_world = []
            category_cams_intr4x4 = []
            category_pts3d = None
            category_pts3d_colors = None
            category_pts3d_normals = None
            for sequence in tqdm(sequences):
                if sequence.category == category:
                    cams_tform4x4_world, cams_intr4x4, cams_imgs = sequence.read_cams(cams_count=imgs_count)

                    category_cams_imgs += cams_imgs
                    category_cams_tform4x4_world.append(torch.stack(cams_tform4x4_world, dim=0))
                    category_cams_intr4x4.append(torch.stack(cams_intr4x4, dim=0))
                    if category_mesh is None:
                        category_pts3d, category_pts3d_colors, category_pts3d_normals = sequence.read_pcl()
                        category_mesh = sequence.read_mesh()

            category_cams_intr4x4 = torch.cat(category_cams_intr4x4, dim=0)
            category_cams_tform4x4_world = torch.cat(category_cams_tform4x4_world, dim=0)
            #category_cams_imgs = torch.cat(category_cams_imgs, dim=0)
            logger.info(f'mesh has {len(category_mesh.verts)} vertices and {len(category_mesh.faces)} faces.')
            show_scene(cams_tform4x4_world=category_cams_tform4x4_world, cams_intr4x4=category_cams_intr4x4,
                       cams_imgs=category_cams_imgs, meshes=[category_mesh], viewpoints_count=viewpoints_count,
                       fpath=Path(f'{category}_frames.webm'), H=H, W=W, pts3d_size=1.)


            #show_scene(cams_tform4x4_world=category_cams_tform4x4_world, cams_intr4x4=category_cams_intr4x4,
            #           cams_imgs=category_cams_imgs, pts3d_colors=[category_pts3d_colors], pts3d=[category_pts3d],
            #           viewpoints_count=9, fpath=f'{category}.png')

    def visualize_category_pcls(self, imgs_count=5, viewpoints_count=16, H=1080, W=1980, ref_count=1):
        sequences = self.get_sequences()
        from od3d.cv.visual.resize import resize
        from od3d.cv.visual.show import show_scene, show_imgs

        categorical_fpaths = {category: [] for category in self.categories}
        for category in tqdm(self.categories):
            for sequence in tqdm(sequences):
                if sequence.category == category:
                    fpath = Path(f'{category}_{sequence.name}_pcl.webm')
                    categorical_fpaths[category].append(fpath)
                    instance_pts3d, instance_pts3d_colors, instance_pts3d_normals = sequence.read_pcl()
                    show_scene(pts3d_colors=[instance_pts3d_colors], pts3d=[instance_pts3d],
                               viewpoints_count=viewpoints_count, fpath=fpath, H=H, W=W)

            from od3d.cv.io import write_webm_videos_side_by_side
            write_webm_videos_side_by_side(in_fpaths=categorical_fpaths[category],
                                           out_fpath=Path(f'{category}_pcl.webm'))

    def visualize_category_meshes(self, viewpoints_count=16, H=1080, W=1980,
                                  modalities=['ncds'], #, 'nn_geo', 'nn_app', 'cycle_weight', 'nn_app_cycle_weight'],
                                  cyclic_weight_temp=0.9, ref_count=1):
        # modalities = ['ncds', 'nn_geo', 'nn_app', 'cycle_weight', 'nn_app_cycle_weight']
        sequences = self.get_sequences()
        from od3d.cv.visual.resize import resize
        from od3d.cv.visual.show import show_scene, show_imgs
        categorical_src_counter = {category: 0 for category in self.categories}
        modality_categorical_ref_fpaths ={ modality: {category: {} for category in self.categories} for modality in modalities}
        for src_sequence in tqdm(sequences):
            categorical_src_counter[src_sequence.category] += 1
            ref_counter = 0
            for ref_sequence in tqdm(sequences):
                if ref_counter >= ref_count and categorical_src_counter[src_sequence.category] > ref_count:
                    break

                if src_sequence.category == ref_sequence.category:
                    src_sequence_mesh = src_sequence.read_mesh()
                    ref_sequence_mesh = ref_sequence.read_mesh()
                    ref_counter += 1
                else:
                    continue
                    #ref_sequence_pts3d, ref_sequence_pts3d_colors, ref_sequence_pts3d_normals = ref_sequence.read_pcl()


                if modalities != ['ncds']:
                    #sequence.preprocess_mesh_feats(override=False)
                    # logger.info(f'mesh is watertight: {sequence_mesh.to_o3d().is_watertight()}')
                    #category_sequence.preprocess_mesh_feats(override=False)
                    #sequence.preprocess_mesh_feats_dist(category_sequence, override=False)
                    mesh_feats_dist = src_sequence.read_mesh_feats_dist(ref_sequence)

                    dist_ref_geo_max = torch.cdist(ref_sequence_mesh.verts[None,], ref_sequence_mesh.verts[None,]).max().detach()  #
                    dist_src_geo_max = torch.cdist(src_sequence_mesh.verts[None,], src_sequence_mesh.verts[None,]).max().detach()  #

                    argmin_ref_from_src = mesh_feats_dist.argmin(dim=-1)  # N,
                    argmin_src_from_ref = mesh_feats_dist.argmin(dim=-2)  # R,
                    src_cyclic_dist = (src_sequence_mesh.verts - src_sequence_mesh.verts[argmin_src_from_ref[argmin_ref_from_src]]).norm(dim=-1,).detach() / dist_src_geo_max  # N,
                    ref_cyclic_dist = (ref_sequence_mesh.verts - ref_sequence_mesh.verts[argmin_ref_from_src[argmin_src_from_ref]]).norm(dim=-1,).detach() / dist_ref_geo_max  # R,

                    from od3d.cv.select import batched_index_select
                    src_cyclic_dist[
                        batched_index_select(input=mesh_feats_dist, index=argmin_ref_from_src[..., None], dim=1).isinf()[:,
                        0]] = src_cyclic_dist.max() # torch.inf
                    ref_cyclic_dist[
                        batched_index_select(input=mesh_feats_dist.T, index=argmin_src_from_ref[..., None], dim=1).isinf()[
                        :, 0]] = ref_cyclic_dist.max() # torch.inf
                    cycle_weight = torch.exp(-((src_cyclic_dist / cyclic_weight_temp)))


                for modality in modalities:
                    add_cbar = False
                    cbar_max = 1.
                    cbar_min = 0.
                    src_sequence_mesh = src_sequence.read_mesh()
                    from od3d.cv.visual.show import dist_to_blue_yellow
                    from od3d.cv.geometry.transform import transf4x4_from_rot3d, transf3d_broadcast
                    _tform4x4 = transf4x4_from_rot3d(torch.Tensor([0., 0, np.pi]))
                    src_sequence_mesh.verts = transf3d_broadcast(pts3d= src_sequence_mesh.verts, transf4x4=_tform4x4)

                    if modality == 'ncds':
                        src_sequence_mesh.rgb = src_sequence_mesh.verts_ncds
                    elif modality == 'nn_app':  #  'nn_geo', 'nn_app', 'cycle_weight', 'nn_app_cycle_weight'
                        src_sequence_mesh.rgb = ref_sequence_mesh.verts_ncds[argmin_ref_from_src].clone()


                    elif modality == 'nn_app_dist':
                        dist = (src_sequence_mesh.verts_ncds - ref_sequence_mesh.verts_ncds[argmin_ref_from_src].clone()).norm(dim=-1) / dist_src_geo_max
                        add_cbar = True
                        cbar_max = dist.max()
                        cbar_min = dist.min()
                        src_sequence_mesh.rgb = dist_to_blue_yellow(dist, normalize=True).permute(1, 0)

                    elif modality == 'nn_geo_dist':
                        dist_geo = torch.cdist(src_sequence_mesh.verts[None,], ref_sequence_mesh.verts[None,]).detach()[0] / dist_src_geo_max
                        dist_geo_min = dist_geo.min(dim=-1).values.clone().clamp(0, 1)
                        # logger.info(dist_geo_min)
                        add_cbar= True
                        cbar_max = dist_geo_min.max()
                        cbar_min = dist_geo_min.min()
                        src_sequence_mesh.rgb = dist_to_blue_yellow(dist_geo_min).permute(1, 0)

                    elif modality == 'nn_cycle':
                        src_sequence_mesh.rgb = src_sequence_mesh.verts_ncds[argmin_src_from_ref[argmin_ref_from_src]].clone()
                    elif modality == 'nn_app_cycle_weight':
                        src_sequence_mesh.rgb = ref_sequence_mesh.verts_ncds[argmin_ref_from_src].clone() * cycle_weight[:, None]
                    elif modality == 'nn_geo':
                        dist_geo = torch.cdist(src_sequence_mesh.verts[None,], ref_sequence_mesh.verts[None,]).detach()[0]
                        src_sequence_mesh.rgb = ref_sequence_mesh.verts_ncds[dist_geo.argmin(dim=-1)].clone()
                    elif modality == 'cycle_dist':
                        add_cbar = True
                        cbar_max = src_cyclic_dist.max()
                        cbar_min = src_cyclic_dist.min()
                        src_sequence_mesh.rgb = dist_to_blue_yellow(src_cyclic_dist).permute(1, 0) # [:, None].repeat(1, 3) * 0.9
                    if modality == 'ncds':
                        fpath = Path(f'{src_sequence.category}_{src_sequence.name}_ncds.webm')
                    else:
                        fpath = Path(f'{src_sequence.category}_src_{src_sequence.name}_ref_{ref_sequence.name}_{modality}.webm')
                        if ref_sequence.name not in modality_categorical_ref_fpaths[modality][src_sequence.category].keys():
                            modality_categorical_ref_fpaths[modality][src_sequence.category][ref_sequence.name] = []

                        if ref_sequence.name == src_sequence.name:
                            modality_categorical_ref_fpaths[modality][src_sequence.category][
                                ref_sequence.name].append(Path(f'{src_sequence.category}_{src_sequence.name}_ncds.webm'))
                        else:
                            modality_categorical_ref_fpaths[modality][src_sequence.category][ref_sequence.name].append(fpath)
                    show_scene(meshes=[src_sequence_mesh], viewpoints_count=viewpoints_count,
                               fpath=fpath, H=H, W=W, add_cbar=add_cbar, cbar_max=cbar_max, cbar_min=cbar_min)

        for modality in modalities:
            for category in tqdm(self.categories):
                for ref_name in modality_categorical_ref_fpaths[modality][category].keys():
                    from od3d.cv.io import write_webm_videos_side_by_side
                    if len(modality_categorical_ref_fpaths[modality][category][ref_name]) > 0:
                        write_webm_videos_side_by_side(in_fpaths=modality_categorical_ref_fpaths[modality][category][ref_name],
                                                       out_fpath=Path(f'{category}_{ref_name}_{modality}.webm'))