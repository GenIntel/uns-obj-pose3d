import logging

import od3d.io

logger = logging.getLogger(__name__)
import shutil
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frame_meta import OD3D_FrameMeta, OD3D_FrameMetaSubsetMixin, \
    OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMetaCategoriesMixin, OD3D_FrameMetaBBoxsMixin
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from od3d.datasets.coco.enum import COCO_CATEGORIES, COCO_SUBSETS
from od3d.io import run_cmd

from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
import pycocotools.coco
import torch
from dataclasses import dataclass
from tqdm import tqdm
import torch.utils.data
import inspect

@dataclass
class COCO_FrameMeta(OD3D_FrameMetaBBoxsMixin, OD3D_FrameMetaCategoriesMixin, OD3D_FrameMetaSubsetMixin,
                     OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMeta):

    @property
    def name_unique(self):
        return f'{self.subset}/{self.name}'

    @staticmethod
    def load_from_raw(image: Dict, annotations: List, subset: str, dict_category_id_to_name: Dict):
        # image keys:  'license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'
        # annotation keys: 'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'
        name = Path(image['file_name']).stem

        rfpath_rgbs = Path(subset)
        rfpath_rgb = rfpath_rgbs.joinpath(image['file_name'])

        H = int(image['height'])
        W = int(image['width'])
        size = torch.Tensor([H, W]).tolist()

        categories = []
        bboxs = []
        for annotation in annotations:
            # logger.info(f'category id {annotation["category_id"]}')
            categories.append(dict_category_id_to_name[annotation['category_id']])
            bboxs.append(annotation['bbox'])

        return COCO_FrameMeta(rfpath_rgb=rfpath_rgb, subset=subset, categories=categories, l_size=size,
                              l_bboxs=bboxs, name=name)


class COCO(OD3D_Dataset):
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[COCO_CATEGORIES]=None,
                 dict_nested_frames: Dict=None, dict_nested_frames_ban: Dict=None,
                 transform=None, index_shift=0, subset_fraction=1.):

        categories = categories if categories is not None else COCO_CATEGORIES.list()
        super().__init__(categories=categories, dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, index_shift=index_shift,
                         subset_fraction=subset_fraction)

    def filter_list_frames_unique(self, list_frames_unique):
        list_frames_unique = super().filter_list_frames_unique(list_frames_unique)
        list_frames_unique_filtered = []
        logger.info('filtering frames categorical...')
        for frame_name_unique in tqdm(list_frames_unique):
            meta = COCO_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=frame_name_unique)
            if len(set(self.categories).intersection(set(meta.categories))) > 0:
                list_frames_unique_filtered.append(frame_name_unique)
        return list_frames_unique_filtered

    def get_item(self, item):
        frame_meta = COCO_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
        return OD3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta, meta=frame_meta, modalities=self.modalities, categories=self.categories)

    @staticmethod
    def setup(config: DictConfig):

        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous COCO")
            shutil.rmtree(path_raw)


        # run_cmd('curl https://sdk.cloud.google.com | bash', logger=logger, live=True)
        subsets_images = ['train2017', 'val2017', 'test2017']
        # 'unlabeled2017'
        for subset in subsets_images:
            path_subset = path_raw.joinpath(subset)
            if path_subset.exists() and not config.setup.override:
                logger.info(f"Found subset of COCO at {path_subset}")
            else:
                path_raw.mkdir(parents=True, exist_ok=True)
                run_cmd(f'cd {path_raw} && wget http://images.cocodataset.org/zips/{subset}.zip', logger=logger, live=True)
                run_cmd(f'cd {path_raw} && unzip {subset}.zip', logger=logger, live=True)
                run_cmd(f'cd {path_raw} && rm {subset}.zip', logger=logger, live=True)

        subsets_annotations = ['annotations_trainval2017', 'image_info_test2017']
        # 'unlabeled2017'
        for subset in subsets_annotations:
            path_subset = path_raw.joinpath(subset)
            if path_subset.exists() and not config.setup.override:
                logger.info(f"Found subset of COCO at {path_subset}")
            else:
                path_raw.mkdir(parents=True, exist_ok=True)
                run_cmd(f'cd {path_raw} && wget http://images.cocodataset.org/annotations/{subset}.zip', logger=logger, live=True)
                run_cmd(f'cd {path_raw} && unzip {subset}.zip', logger=logger, live=True)
                run_cmd(f'cd {path_raw} && rm {subset}.zip', logger=logger, live=True)
            # path_subset = path_raw.joinpath(subset)
            # path_subset.mkdir(parents=True, exist_ok=True)
            # run_cmd(f'gsutil -m rsync gs://images.cocodataset.org/{subset} {subset}', logger=logger, live=True)

    @staticmethod
    def extract_meta(config: DictConfig):

        path_raw = Path(config.path_raw)
        path_meta = COCO.get_path_meta(config=config)

        for subset in ['val2017', 'train2017']: # 'train2017', 'val2017'
            annFile = path_raw.joinpath('annotations', f'instances_{subset}.json')  #f'{dataDir}/annotations/instances_{dataType}.json'
            coco = pycocotools.coco.COCO(annFile)

            cats = coco.loadCats(coco.getCatIds())
            dict_category_id_to_name = {cat['id']: cat['name'] for cat in cats}
            all_categories = [cat['name'] for cat in cats]
            #logger.info("Categories:", all_categories)
            logger.info(f'Found {len(all_categories)} number of categories.')

            imgIds = coco.getImgIds()
            images = coco.loadImgs(imgIds)
            logger.info(f"Number of images: {len(images)}")

            for i in tqdm(range(len(images))):
                image = images[i] # image in images:
                image_id = image['id']
                annIds = coco.getAnnIds(imgIds=image_id)
                annotations = coco.loadAnns(annIds)
                frame_meta = COCO_FrameMeta.load_from_raw(image=image, subset=subset, annotations=annotations, dict_category_id_to_name=dict_category_id_to_name)
                if frame_meta is not None:
                    frame_meta.save(path_meta=path_meta)

    def preprocess(self, override: False):
        pass