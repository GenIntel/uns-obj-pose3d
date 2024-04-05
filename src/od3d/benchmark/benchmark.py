import logging
logger = logging.getLogger(__name__)
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path
from od3d.datasets.dataset import OD3D_Dataset
from od3d.methods.method import OD3D_Method
import numpy as np
import random
import torch


class OD3D_Benchmark:
    def __init__(self, config: DictConfig):
        self.config = config

        self.logging_dir = Path(self.config.logger.local_dir).joinpath(self.config.run_name)
        self.logging_dir.mkdir(parents=True)
        if self.config.logger.use_wandb:
            wandb.login()
            wandb.init(project=self.config.logger.wandb_project_name, config=OmegaConf.to_container(self.config, resolve=True),
                       dir=Path(self.config.logger.local_dir), name=self.config.run_name, reinit=True)

    def run(self):
        random.seed(self.config.get('seed_number', 0))
        np.random.seed(self.config.get('seed_number', 0))
        torch.manual_seed(self.config.get('seed_number', 0))

        # 1. setup datasets
        datasets_val = {}
        if "val_datasets" in self.config.keys():
            for dataset_val_key in self.config.val_datasets.keys():
                if self.config.val_datasets[dataset_val_key].get('skip', False):
                    continue
                if self.config.val_datasets[dataset_val_key] is None or self.config.val_datasets[dataset_val_key].get('name', None) is None:
                    continue

                logger.info(f'create val dataset {self.config.val_datasets[dataset_val_key].name}')
                datasets_val[dataset_val_key] = OD3D_Dataset.subclasses[self.config.val_datasets[dataset_val_key].class_name].create_from_config(config=self.config.val_datasets[dataset_val_key])

        datasets_test = []
        for dataset_test_key in self.config.test_datasets.keys():
            if self.config.test_datasets[dataset_test_key].get('skip', False):
                continue
            logger.info(f'create test dataset {self.config.test_datasets[dataset_test_key].name}')

            datasets_test.append(
                OD3D_Dataset.subclasses[self.config.test_datasets[dataset_test_key].class_name].create_from_config(
                    config=self.config.test_datasets[dataset_test_key]))

        datasets_train = {}
        for dataset_train_key in self.config.train_datasets.keys():
            logger.info(f'create train dataset {self.config.train_datasets[dataset_train_key].name}')
            datasets_train[dataset_train_key] = OD3D_Dataset.subclasses[self.config.train_datasets[dataset_train_key].class_name].create_from_config(config=self.config.train_datasets[dataset_train_key])


        # 2. setup method
        method = OD3D_Method.subclasses[self.config.method.class_name](self.config.method, logging_dir=self.logging_dir)

        # 3. train method
        if self.config.train:
            logger.info('train')
            method.train(datasets_train, datasets_val)

        # 4. test method (logs results inside class)
        if self.config.test:

            logger.info('test')
            for dataset_test in datasets_test:
                results_test = method.test(dataset_test)
                results_test.log_with_prefix(prefix=f'test/{dataset_test.name}')