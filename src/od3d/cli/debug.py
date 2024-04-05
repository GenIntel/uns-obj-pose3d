import logging
from pathlib import Path
logger = logging.getLogger(__name__)
import typer
import od3d.io
import re

app = typer.Typer()

@app.command()
def latest_logs():
    logging.basicConfig(level=logging.DEBUG)
    cfg = od3d.io.load_hierarchical_config()
    logs = {}
    for p in Path(cfg.platform.path_logs).iterdir():
        if re.match('[^.]+.o[0-9]+', p.name) is not None:
            with open(p) as f:
                logs[p.stat().st_mtime] = f'{p.name} \n\n' + ''.join(f.readlines()) + f'\n\n {p.name}'
    logs_sorted = sorted(logs.items(), key=lambda k: k[0])

    latest_logs_name = logs_sorted[-1][0]
    latest_logs_content = logs_sorted[-1][1]
    logger.info(latest_logs_name)
    logger.info(latest_logs_content)
    # bench multiple -p torque
@app.command()
def hello_world():
    logging.basicConfig(level=logging.DEBUG)
    logger.info("hello world")