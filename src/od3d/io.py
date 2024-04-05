import logging
logger = logging.getLogger(__name__)
import urllib.request
import time
import sys
from pathlib import Path
import tarfile
import zipfile
import shutil
import os
import gdown
from hydra import compose, initialize, initialize_config_dir

from omegaconf import DictConfig, OmegaConf
import json
from typing import Dict
import subprocess

import importlib


def is_fpath_video(fpath: Path):
    return fpath.suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']

def is_fpath_image(fpath: Path):
    return fpath.suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
def download(url: str, fpath: Path):
    if fpath.exists():
        logging.warning(f"File {fpath} already exists. Skip download {url}.")
    else:
        if not fpath.parent.exists():
            fpath.parent.mkdir(parents=True)

        if "google.com" in url:
            gdown.download(url=url, output=str(fpath), fuzzy=True)
        else:
            urllib.request.urlretrieve(url, fpath, reporthook)


def unzip(fpath: Path, dst: Path):
    with zipfile.ZipFile(fpath, 'r') as zip_ref:
        zip_ref.extractall(dst)
    os.remove(fpath)

def untar(fpath: Path, dst: Path):
    file = tarfile.open(fpath)
    file.extractall(dst)
    file.close()
    os.remove(fpath)

def move_dir(src: Path, dst: Path):
    for _fpath in src.iterdir():
        shutil.move(_fpath, dst)
    shutil.rmtree(src)

def rm_dir(path: Path):
    try:
        logger.info(f'removing directory {path}')
        path.unlink()
        shutil.rmtree(path)
    except Exception as e:
        logger.warning(e)

from tqdm import tqdm
from omegaconf import open_dict
import multiprocessing

def load_single_hierarchical_config(procnum, return_dict, config_dir_rel, benchmark, platform, ablations, overrides):
    """worker function"""
    logger.info(f'start process {procnum}')
    return_dict[procnum] = procnum
    with initialize(version_base=None, config_path=config_dir_rel, job_name="test_app"):
        overrides = [f"+ablations/{Path(ablation).parent}={Path(ablation).stem}" for ablation in ablations] + [
            "platform=" + platform] + overrides
        cfg = compose(config_name=benchmark, overrides=overrides)

        with open_dict(cfg):
            cfg.ablation_name = '_'.join([ablation.stem for ablation in ablations])

        logger.info(cfg.ablation_name)
        return_dict[procnum] = cfg

def load_multiple_hierarchical_configs(benchmark="defaults", platform="local", multiple_ablations=[]):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    config_dir_rel = "../../config"
    cfgs = []
    overrides = []
    for a, ablations in tqdm(enumerate(multiple_ablations)):
        p = multiprocessing.Process(target=load_single_hierarchical_config, args=(a, return_dict, config_dir_rel, benchmark, platform, ablations,
                                                                                  overrides))
        jobs.append(p)
        p.start()

    for a, proc in enumerate(jobs):
        proc.join()
        cfgs.append(return_dict[a])
    return cfgs

def load_hierarchical_config(benchmark="defaults", platform="local", ablations=[], overrides=[]):
    config_dir_rel = "../../config"
    with initialize(version_base=None, config_path=config_dir_rel, job_name="test_app"):
        overrides = [f"+ablations/{Path(ablation).parent}={Path(ablation).stem}" for ablation in ablations] + ["platform=" + platform] + overrides
        cfg = compose(config_name=benchmark, overrides=overrides)

        #if ablations is None:
        #    cfg = compose(config_name=benchmark, overrides=["platform=" + platform] + overrides)
        #else:
        #    cfg = compose(config_name=benchmark, overrides=["+ablations=" + ablation, "platform=" + platform] + overrides)
    return cfg

def read_config_intern(rfpath: Path, benchmark="defaults", platform="local", overrides=[]):
    config_dir_rel = "../../config"

    try:
        with initialize(version_base=None, config_path=config_dir_rel, job_name="test_app"):
            cfg = compose(config_name=benchmark, overrides=[f"+{rfpath.parent}=" + str(rfpath.stem), "platform=" + platform] + overrides)

        cfg = cfg
        for key in str(rfpath.parent).split('/'):
            cfg = cfg.get(key)
    except hydra.errors.ConfigCompositionException as e:
        fpath = Path("config").joinpath(rfpath)
        logger.warning(e)
        logger.warning('trying to read with OmegaConf')
        cfg = OmegaConf.load(fpath)

    return cfg


import hydra.errors

def read_config_extern(fpath: Path):
    try:
        with initialize_config_dir(config_dir=str(fpath.parent.absolute()), job_name="test_app_extern"):
            cfg = compose(config_name=fpath.stem)
    except hydra.errors.ConfigCompositionException as e:
        logger.warning(e)
        logger.warning('trying to read with OmegaConf')
        cfg = OmegaConf.load(fpath)

    return cfg

def write_config_to_json_file(config: DictConfig, fpath: Path):
    write_json(config=dict(config), fpath=fpath)

def write_json(config: Dict, fpath: Path):
    fpath.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(fpath.expanduser(), "w") as outfile:
        json.dump(config, outfile)

def read_json(fpath: Path):
    with open(fpath.expanduser(), 'r') as openfile:
        config = json.load(openfile)
    return config

def read_yaml(fpath: Path, resolve=True):
    cfg = OmegaConf.load(fpath)
    cfg = OmegaConf.to_container(cfg, resolve=resolve)
    return cfg

def run_cmd(cmd, logger, live=False, background=False):
    if logger is not None:
        logger.info(f'Run command {cmd}')
    if live:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)

        for line in process.stdout:
            # Print or process the live output as needed
            logger.info(line)
            # print(line, end='')

        # Wait for the subprocess to complete
        process.wait()

        # Retrieve the return code of the subprocess
        return_code = process.returncode
        logger.info(f'Return Code {return_code}')

    else:
        if not background:
            res = subprocess.run(cmd, capture_output=True, shell=True)
            if logger is not None:
                logger.info(res.stdout.decode("utf-8"))
                logger.info(res.stderr.decode("utf-8"))
            return res.stdout.decode("utf-8")
        else:
            #child = RunCmdBackgroundProcess(cmd, os.getpid())
            from multiprocessing import Process
            pid = os.getpid()
            child_proc = Process(target=run_child, args=(cmd, pid,))
            child_proc.daemon = True
            child_proc.start()

def run_child(cmd, parent_pid):
    """
    Start a child process by running self._cmd.
    Wait until the parent process (self._parent) has died, then kill the
    child.
    """
    import psutil
    from time import sleep
    _parent = psutil.Process(pid=parent_pid)
    _child = subprocess.Popen(cmd, shell=True)
    try:
        #with open("log.txt", "a") as myfile:
        #    myfile.write(_parent.status())
        while _parent.status() == psutil.STATUS_RUNNING or _parent.status() == psutil.STATUS_SLEEPING:
            sleep(1)
    except psutil.NoSuchProcess:
        pass
    finally:
        _child.terminate()

def read_str_from_file(fpath: Path):
    with open(fpath, 'r') as file:
        data = file.read().rstrip()
    return data

def write_str_to_file(fpath: Path, text: str):
    with open(fpath, "w") as file:
        file.write(text)

from typing import List
from copy import deepcopy
from enum import Enum
import numpy as np
import torch


def write_dict_as_yaml(fpath: Path, _dict: Dict, save_enum_as_str=False):
    if save_enum_as_str:
        _dict = {key: str(value) if isinstance(value, Enum) else value for key, value in deepcopy(_dict).items()}

    conf = OmegaConf.create(_dict)
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, 'w') as fp: #  tempfile.NamedTemporaryFile()
        OmegaConf.save(config=conf, f=fp.name)
def read_dict_from_yaml(fpath: Path):
    with open(fpath, 'r') as fp:
        loaded = OmegaConf.load(fp.name)
    return loaded

# import pyarrow as pa
# import pyarrow.parquet as pq
# def save_dict_as_pandas_df(fpath: Path, _dict: Dict):
#
#     # Convert PyTorch tensors to NumPy arrays
#     data_np = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in _dict.items()}
#
#     # Convert NumPy arrays to PyArrow arrays
#     arrays = {key: pa.array(value) if isinstance(value, np.ndarray) else value for key, value in data_np.items()}
#
#     # Create a PyArrow Table from the arrays
#     table = pa.Table.from_pydict(arrays)
#
#     # Write the table to a Parquet file
#     pq.write_table(table, fpath)
#
# def load_dict_from_parquet(fpath):
#     # Read the Parquet file into a PyArrow Table
#     table = pq.read_table(fpath)
#     # Access the schema of the table
#     schema = table.schema
#
#     # Convert PyArrow arrays to NumPy arrays
#     arrays = {column.name: column.to_numpy() if schema.field(column.name).type == pa.Array else column for column in table.columns}
#
#     # Convert NumPy arrays to PyTorch tensors
#     data = {key: torch.tensor(value) if isinstance(value, np.array) else value for key, value in arrays.items()}
#
#     return data

def write_list_as_yaml(fpath: Path, _list: List[str]):
    conf = OmegaConf.create(_list)
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, 'w') as fp: #  tempfile.NamedTemporaryFile()
        OmegaConf.save(config=conf, f=fp.name)

def read_list_from_yaml(fpath: Path):
    with open(fpath, 'r') as fp:
        loaded = OmegaConf.load(fp.name)
    return loaded


def get_obj_from_config(*args, config: DictConfig, **kwargs):
    class_name_split = config.class_name.split('.')
    module_name = '.'.join(class_name_split[:-1])
    class_name = class_name_split[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(*args, **{**kwargs, **config.kwargs})
