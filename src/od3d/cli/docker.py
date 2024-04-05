import logging
from pathlib import Path
logger = logging.getLogger(__name__)
from od3d.io import run_cmd
import typer
app = typer.Typer()

@app.command()
def od3d():
    logging.basicConfig(level=logging.INFO)
    cmd = 'docker build -f docker/Dockerfile -t limpbot/od3d:v1 --build-arg UID=$(id -u) --build-arg GID=$(id DD-g) .'
    run_cmd(cmd=cmd, logger=logger, live=True)

@app.command()
def zsp():
    logging.basicConfig(level=logging.INFO)
    cmd = 'docker build -f third_party/envs/Zero-Shot-Pose/Dockerfile -t limpbot/zsp:v1 third_party/zero-shot-pose'
    run_cmd(cmd=cmd, logger=logger, live=True)

def get_cmd_zsp_run(gpus: str = 'all', port: str = 5000):
    return f'docker run --gpus device={gpus} -p {port}:5000 -t limpbot/zsp:v1'

def get_cmd_zsp_stop(port: str = 5000):
    return f'docker container stop $(docker container ps --filter "ancestor=limpbot/zsp:v1" --filter "publish={port}" -q)'

@app.command()
def zsp_run(gpus: str = typer.Option('all', '-g', '--gpus'),
            port: str = typer.Option(5000, '-p', '--port')):
    logging.basicConfig(level=logging.INFO)
    cmd = get_cmd_zsp_run(gpus=gpus, port=port)
    run_cmd(cmd=cmd, logger=logger, live=False, background=True)

@app.command()
def zsp_stop(port: str = typer.Option(5000, '-p', '--port')):
    logging.basicConfig(level=logging.INFO)
    cmd = get_cmd_zsp_stop(port=port)
    run_cmd(cmd=cmd, logger=logger, live=True)

@app.command()
def droid_slam():
    logging.basicConfig(level=logging.INFO)
    cmd = 'docker build -f third_party/envs/DROID-SLAM/Dockerfile -t limpbot/droid-slam:v1 --build-arg UID=$(id -u) --build-arg GID=$(id -g) third_party/DROID-SLAM'
    run_cmd(cmd=cmd, logger=logger, live=True)
