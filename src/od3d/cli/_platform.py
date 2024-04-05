
import logging
from pathlib import Path
logger = logging.getLogger(__name__)
import typer
import od3d.io
import re
from pygit2 import Repository

app = typer.Typer()
from od3d.benchmark.run import torque_run_method_or_cmd, slurm_run_method_or_cmd
from omegaconf import open_dict
from od3d.io import run_cmd
import subprocess

@app.command()
def slurm():
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='slurm')

    logger.info(f'ssh slurm')
    logger.info(f'cd {config.platform.path_od3d} && source venv310/bin/activate')

@app.command()
def slurm_run(cmd: str = typer.Option('od3d debug hello-world', '-c', '--command'),):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='slurm')
    with open_dict(config):
        config.branch = Repository('.').head.shorthand  # 'master'
    slurm_run_method_or_cmd(config, cmd)

@app.command()
def rsync_configs(platform: str = typer.Option(None, '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config_target = od3d.io.load_hierarchical_config(platform=platform)
    config_source = od3d.io.load_hierarchical_config(platform='local')

    source_link = f'{config_source.platform.link}:' if config_source.platform.link != 'local' else ''
    target_link = f'{config_target.platform.link}:' if config_target.platform.link != 'local' else ''

    config_rfpaths_source = ['credentials/default.yaml',
                             f'platform/{platform}.yaml']
    config_rfpaths_target = ['credentials/default.yaml',
                             f'platform/local.yaml']

    for config_rfpath_source, config_rfpath_target in zip(config_rfpaths_source, config_rfpaths_target):
        path_source = Path(config_source.platform.path_od3d).joinpath('config', config_rfpath_source)
        path_target = Path(config_target.platform.path_od3d).joinpath('config', config_rfpath_target)
        od3d.io.run_cmd(cmd=f'rsync -avrzP {source_link}{path_source} {target_link}{path_target}', live=True, logger=logger)

@app.command()
def run(platform: str = typer.Option(None, '-p', '--platform'),
        cmd: str = typer.Option('od3d debug hello-world', '-c', '--command')):
    logging.basicConfig(level=logging.INFO)

    # run_cmd(f'od3d platform rsync-configs -p {platform}', logger=logger)

    config = od3d.io.load_hierarchical_config(platform=platform)

    with open_dict(config):
        config.branch = Repository('.').head.shorthand  # 'master'

    if platform == 'slurm':
        slurm_run_method_or_cmd(config, cmd)
    elif platform == 'torque':
        torque_run_method_or_cmd(config, cmd)
    else:
        raise NotImplementedError


def get_slurm_jobs_ids(job_id_treshold=None):
    slurm_result = subprocess.run(f'ssh slurm "squeue --me"', capture_output=True, shell=True)
    slurm_jobs = slurm_result.stdout.decode("utf-8").split("\n")
    slurm_jobs_ids = []
    for slurm_job in slurm_jobs[1:]:
        slurm_job_split = slurm_job.split()
        if len(slurm_job_split) > 0:
            slurm_jobs_ids.append(int(slurm_job_split[0]))
    if job_id_treshold is not None:
        slurm_jobs_ids = list(filter(lambda job_id: job_id < job_id_treshold, slurm_jobs_ids))
    return slurm_jobs_ids

def get_torque_jobs_ids(job_id_treshold=None):
    torque_result = subprocess.run(f"ssh torque 'qstat -a -u $(whoami)'", capture_output=True, shell=True)
    torque_jobs = torque_result.stdout.decode("utf-8").split("\n")
    torque_jobs_ids = [int(torque_job.split(".")[0]) for torque_job in torque_jobs[5:] if len(torque_job) > 0]

    if job_id_treshold is not None:
        torque_jobs_ids = list(filter(lambda job_id: job_id < job_id_treshold, torque_jobs_ids))

    return torque_jobs_ids
@app.command()
def rm_installing_txt(platform: str = typer.Option(None, '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform)
    run_cmd(f"ssh {platform} 'rm {config.platform.path_od3d}/installing.txt'", logger=logger)

@app.command()
def rm_git_lock(platform: str = typer.Option(None, '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform)
    run_cmd(f"ssh {platform} 'rm {config.platform.path_od3d}/installing.txt'", logger=logger)


@app.command()
def stop(platform: str = typer.Option(None, '-p', '--platform'),
         job: str = typer.Option(None, '-j', '--job')):
    if platform == 'torque':
        logging.basicConfig(level=logging.INFO)

        jobs = job.split(',')
        for job in jobs:
            if job.startswith('l'):
                torque_jobs_ids = get_torque_jobs_ids(int(job[1:]))
                logger.info(f'stop torque job ids {torque_jobs_ids}')
            else:
                logger.info(f'stop torque job ids {job}')
                torque_jobs_ids = [int(job)]

            stop_torque_jobs_ids(torque_jobs_ids)


    elif platform == 'slurm':
        logging.basicConfig(level=logging.INFO)

        jobs = job.split(',')
        for job in jobs:
            if job.startswith('l'):
                slurm_jobs_ids = get_slurm_jobs_ids(int(job[1:]))
                logger.info(f'stop slurm job ids {slurm_jobs_ids}')
            else:
                slurm_jobs_ids = [int(job)]

            stop_slurm_jobs(slurm_jobs_ids)
    else:
        raise NotImplementedError


def stop_slurm_jobs(slurm_jobs_ids):
    for job_id in slurm_jobs_ids:
        slurm_result = subprocess.run(f'ssh slurm "scancel {str(job_id)}"', capture_output=True, shell=True)
        for line in slurm_result.stdout.decode("utf-8").split("\n"):
            logger.info(line)

def stop_torque_jobs_ids(torque_jobs_ids):
    for job_id in torque_jobs_ids:
        torque_result = subprocess.run(f'ssh torque "qdel {job_id}"', capture_output=True, shell=True)
        for line in torque_result.stdout.decode("utf-8").split("\n"):
            logger.info(line)
@app.command()
def stop_not_running(platform: str = typer.Option(None, '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    from od3d.cli.benchmark import get_runs
    runs_running_online = get_runs(state='running')

    if platform == 'slurm':
        # 60j = 60 characters
        format = '"%.18i %.9P %.60j %.8u %.8T %.10M %.9l %.6D %R"'
        slurm_result = subprocess.run(f"ssh slurm 'squeue --me --format={format}'", capture_output=True, shell=True)
        slurm_jobs = slurm_result.stdout.decode("utf-8").split("\n")[1:-1]
        jobs_names_partial = [slurm_job.split()[2] for slurm_job in slurm_jobs]
        jobs_ids = [int(slurm_job.split()[0]) for slurm_job in slurm_jobs]
    else:
        raise NotImplementedError

    jobs_ids_not_running = []
    for j in range(len(jobs_ids)):
        partial_length = len(jobs_names_partial[j])
        runs_running_online_partial = [run_running_online.name[:partial_length] for run_running_online in runs_running_online]
        if jobs_names_partial[j] not in runs_running_online_partial:
            logger.info(f'job {jobs_names_partial[j]} not running.')
            jobs_ids_not_running.append(jobs_ids[j])

    if platform == 'slurm':
        stop_slurm_jobs(jobs_ids_not_running)


@app.command()
def status(platform: str = typer.Option(None, '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    if platform == 'slurm':
        # 60j = 60 characters
        format = '"%.18i %.9P %.60j %.8u %.8T %.10M %.9l %.6D %R"'
        slurm_result = subprocess.run(f"ssh slurm 'squeue --me --format={format}'", capture_output=True, shell=True)
        slurm_jobs = slurm_result.stdout.decode("utf-8").split("\n")
        slurm_jobs_columns = slurm_jobs[0]
        slurm_jobs = slurm_jobs[1:-1]
        import numpy as np
        slurm_jobs_ids = np.array([int(slurm_job.split()[0]) for slurm_job in slurm_jobs])
        slurm_jobs_ids = slurm_jobs_ids.argsort()
        slurm_jobs = [slurm_jobs[id] for id in slurm_jobs_ids]
        logger.info(slurm_jobs_columns)
        for i, slurm_job in enumerate(slurm_jobs):
            logger.info(f'{i}: {slurm_job}')
    elif platform == 'torque':
        torque_result = subprocess.run(f"ssh torque 'qstat -a -u $(whoami)'", capture_output=True, shell=True)
        torque_jobs = torque_result.stdout.decode("utf-8").split("\n")
        for i, torque_job in enumerate(torque_jobs):
            logger.info(f'{i}: {torque_job}')
    else:
        raise NotImplementedError


# sinfo
@app.command()
def torque():
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='torque')

    logger.info(f'ssh torque')
    logger.info(f'cd {config.platform.path_od3d} && source venv310/bin/activate')

