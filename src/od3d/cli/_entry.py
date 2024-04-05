import typer
app = typer.Typer()
from od3d.cli.benchmark import app as app_run
from od3d.cli.dataset import app as app_setup
from od3d.cli.debug import app as app_debug
from od3d.cli.pcl import app as app_pcl
from od3d.cli._platform import app as app_platform
from od3d.cli.table import app as app_table
from od3d.cli.figure import app as app_figure
from od3d.cli.docker import app as app_docker
from od3d.cli.write import app as app_write
from od3d.cli.result import app as app_result

app.add_typer(app_run, name='bench')
app.add_typer(app_setup, name='dataset')
app.add_typer(app_debug, name='debug')
app.add_typer(app_pcl, name='pcl')
app.add_typer(app_platform, name='platform')
app.add_typer(app_table, name='table')
app.add_typer(app_figure, name='figure')
app.add_typer(app_docker, name='docker')
app.add_typer(app_write, name='write')
app.add_typer(app_result, name='result')

def main():
    app()

if __name__ == "__main__":
    main()



