import logging
from pathlib import Path
logger = logging.getLogger(__name__)
import typer
import od3d.io
import re

app = typer.Typer()

@app.command()
def error_choices(platform: str = typer.Option('local', '-p', '--platform'),
                  run: str = typer.Option(None, '-r', '--run'),
                  dataset_rpath: str = typer.Option('test/pascal3d_test', '-d', '--dataset'),
                  choices: str = typer.Option(None, '-c', '--choices'),
                  metric: str = typer.Option('rot_diff_rad', '-m', '--metric'),
                  lt: str = typer.Option(None, '-l', '--lower-than'),
                  gt: str = typer.Option(None, '-g', '--greater-than')):

    # test/objectnet3d_test
    logging.basicConfig(level=logging.INFO)

    choices = choices.split(',')
    if choices is None:
        raise ValueError('Please provide the choices...')

    config = od3d.io.load_hierarchical_config(platform=platform)
    from od3d.io import read_dict_from_yaml
    from od3d.benchmark.results import OD3D_Results

    run_path = Path(config.platform.path_exps).joinpath(run)
    results = OD3D_Results.read_from_local(logging_dir=run_path, dataset_rpath=Path(dataset_rpath))

    import math
    import torch
    list_names_unique = results['name_unique']
    list_filter = torch.ones(len(list_names_unique), dtype=torch.bool)
    if lt is not None:
        list_filter = results[metric] < eval(lt)
    if gt is not None:
        list_filter = results[metric] > eval(gt)
    list_names_unique = [item for item, flag in zip(list_names_unique, list_filter) if flag]
    # list_names_unique = list_names_unique[:10]
    images_rfpaths = [name_unique + '.png' for name_unique in list_names_unique]
    images_root_path = run_path.joinpath(dataset_rpath, 'visual', 'pred_vs_gt_verts_ncds_in_rgb')

    root = tk.Tk()
    app = SurveyApp(root, images_root_path=images_root_path, images_rfpaths=images_rfpaths, choices=choices)
    root.mainloop()

    #from od3d.datasets.dataset import OD3D_Dataset
    #config_dataset = read_dict_from_yaml(run_path.joinpath(dataset_rpath, 'config.yaml'))
    #dict_nested_frames = results['name_unique'].enroll()
    #config_dataset.dict_nested_frames = dict_nested_frames
    #dataset = OD3D_Dataset.subclasses[config_dataset.class_name].create_from_config(config=config_dataset)




import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from typing import List
class SurveyApp:
    def __init__(self, master, images_root_path: Path, images_rfpaths: List[Path],
                 choices: List[str]):
        self.master = master
        self.master.title("Survey")
        self.images_root_path = images_root_path
        self.images_rfpaths = images_rfpaths
        #self.images = ["image1.jpg", "image2.jpg", "image3.jpg"]  # List of image filenames
        self.current_image_index = 0
        self.choices = choices
        self.choices_selected = {choice: [] for choice in choices}

        self.load_image()
        self.create_widgets()

    def load_image(self):
        image_path = self.images_root_path.joinpath(self.images_rfpaths[self.current_image_index])
        self.image = Image.open(image_path)
        ratio = self.image.width / self.image.height
        self.image = self.image.resize((int(600 * ratio), 600)) # , Image.ANTIALIAS
        self.photo = ImageTk.PhotoImage(self.image)

    def create_widgets(self):
        self.image_label = tk.Label(self.master, image=self.photo)
        self.image_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.choices_var = {choice: tk.BooleanVar() for choice in self.choices}
        self.choices_checkbuttons = {} #  {choice:  for choice in self.choices}


        for i, choice in enumerate(self.choices):
            self.choices_checkbuttons[choice] = ttk.Checkbutton(self.master, text=choice, variable=self.choices_var[choice])
            self.choices_checkbuttons[choice].grid(row=i+1, column=0, padx=10, pady=5, sticky="w")

        self.next_button = ttk.Button(self.master, text="Next", command=self.next_image)
        self.next_button.grid(row=len(self.choices)+1, column=0, columnspan=2, padx=10, pady=10)

    def next_image(self):
        # Save selected choices
        for choice in self.choices:
            if self.choices_var[choice].get():
                self.choices_selected[choice].append(True)
            else:
                self.choices_selected[choice].append(False)

        # Clear previous selections
        for choice in self.choices:
            self.choices_var[choice].set(False)

        # Load next image
        self.current_image_index += 1
        if self.current_image_index < len(self.images_rfpaths):
            self.load_image()
            self.image_label.configure(image=self.photo)
        else:
            # End of survey
            self.show_results()

    def show_results(self):
        # Display selected choices
        logger.info("Selected Choices:")
        for choice in self.choices_selected:
            logger.info(f'{choice}: {100 * sum(self.choices_selected[choice]) / len(self.choices_selected[choice])} %')
        self.master.destroy()  # Close the window