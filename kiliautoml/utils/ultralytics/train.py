import math
import os
import re
import shutil
import subprocess
from datetime import datetime
from functools import reduce
from typing import Dict, List

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from kiliautoml.utils.constants import ModelFrameworkT
from kiliautoml.utils.download_assets import download_project_images
from kiliautoml.utils.helpers import categories_from_job, kili_print
from kiliautoml.utils.path import ModelRepositoryDirT
from kiliautoml.utils.type import JobT
from kiliautoml.utils.ultralytics.constants import ULTRALYTICS_REL_PATH, YOLOV5_REL_PATH

env = Environment(
    loader=FileSystemLoader(os.path.abspath(ULTRALYTICS_REL_PATH)),
    autoescape=select_autoescape(),
)


class AutoMLYoloException(Exception):
    pass


def get_output_path_bbox(title: str, path: str, model_framework: str) -> str:
    """Output folder path for the model."""
    # wandb splits the output_path according to "/" and uses the title as name of the project.
    model_output_path = os.path.join(
        path,
        "model",
        model_framework,
        datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        title,  # <-- name of the project in wandb
    )
    illegal_caracters = [",", "#", "?", "%", ":"]
    for char in illegal_caracters:
        model_output_path = model_output_path.replace(char, "_")
    kili_print("output_path of the model:", model_output_path)
    return model_output_path


def yaml_preparation(
    *,
    data_path: str,
    class_names: List[str],
    kili_api_key: str,
    assets,
):

    print("Downloading datasets from Kili")
    train_val_proportions = [0.8, 0.1]
    path = data_path
    if "/kili/" not in path:
        raise ValueError("'path' field in config must contain '/kili/'")

    n_train_assets = math.floor(len(assets) * train_val_proportions[0])
    n_val_assets = math.floor(len(assets) * train_val_proportions[1])
    assets_splits = {
        "train": assets[:n_train_assets],
        "val": assets[n_train_assets : n_train_assets + n_val_assets],
        "test": assets[n_train_assets + n_val_assets :],
    }

    for name_split, assets_split in assets_splits.items():
        if len(assets_split) == 0:
            raise Exception("No asset in dataset, exiting...")
        path_split = os.path.join(path, "images", name_split)

        download_project_images(
            api_key=kili_api_key,
            assets=assets_split,
            output_folder=path_split,
        )

        names = class_names
        path_labels = re.sub("/images/", "/labels/", path_split)
        print(path_labels)
        os.makedirs(path_labels, exist_ok=True)
        for asset in assets_split:
            if asset["labels"]:
                asset_id = asset["id"] + ".txt"  # type: ignore
                with open(os.path.join(path_labels, asset_id), "w") as handler:
                    json_response = asset["labels"][0]["jsonResponse"]
                    for job in json_response.values():
                        save_annotations_to_yolo_format(names, handler, job)


def save_annotations_to_yolo_format(names, handler, job):
    for annotation in job.get("annotations", []):
        name = annotation["categories"][0]["name"]
        try:
            category = names.index(name)
        except ValueError:
            pass
        bounding_poly = annotation.get("boundingPoly", [])
        if len(bounding_poly) < 1:
            continue
        if "normalizedVertices" not in bounding_poly[0]:
            continue
        normalized_vertices = bounding_poly[0]["normalizedVertices"]
        x_s = [vertice["x"] for vertice in normalized_vertices]
        y_s = [vertice["y"] for vertice in normalized_vertices]
        x_min, y_min = min(x_s), min(y_s)
        x_max, y_max = max(x_s), max(y_s)
        _x_, _y_ = (x_max + x_min) / 2, (y_max + y_min) / 2
        _w_, _h_ = x_max - x_min, y_max - y_min
        handler.write(f"{category} {_x_} {_y_} {_w_} {_h_}\n")  # type: ignore


def ultralytics_train_yolov5(
    *,
    api_key: str,
    assets,
    model_repository_dir: ModelRepositoryDirT,
    job: JobT,
    json_args: Dict,  # type: ignore
    epochs: int,
    model_framework: ModelFrameworkT,
    title: str,
    disable_wandb: bool,
    clear_dataset_cache: bool,
    batch_size: int,
) -> float:
    yolov5_path = os.path.join(os.getcwd(), YOLOV5_REL_PATH)

    template = env.get_template("kili_template.yml")
    class_names = categories_from_job(job)
    data_path = os.path.join(model_repository_dir, "data")
    config_data_path = os.path.join(yolov5_path, "data", "kili.yaml")

    if clear_dataset_cache and os.path.exists(data_path) and os.path.isdir(data_path):
        kili_print("Dataset cache for this project is being cleared.")
        shutil.rmtree(data_path)

    model_output_path = get_output_path_bbox(title, model_repository_dir, model_framework)
    os.makedirs(model_output_path, exist_ok=True)

    os.makedirs(os.path.dirname(config_data_path), exist_ok=True)
    yaml_preparation(
        data_path=data_path,
        class_names=class_names,
        kili_api_key=api_key,
        assets=assets,
    )

    with open(config_data_path, "w") as f:
        f.write(
            template.render(
                data_path=data_path,
                class_names=class_names,
                number_classes=len(class_names),
            )
        )

    if not json_args:
        json_args = {}
    json_args["epochs"] = epochs
    args_from_json = reduce(lambda x, y: x + y, ([f"--{k}", f"{v}"] for k, v in json_args.items()))
    kili_print("Starting Ultralytics' YoloV5 ...")
    try:
        yolo_env = os.environ.copy()
        if disable_wandb:
            yolo_env["WANDB_DISABLED"] = "false"
        args = [
            "python",
            "train.py",
            "--data",
            "kili.yaml",
            "--project",
            f"{model_output_path}",
            "--upload_dataset",  # wandb
            "--weights",
            "yolov5n.pt",
            "--batch",
            str(batch_size),
            *args_from_json,
        ]
        print("Executing Yolo with command line:", " ".join(args))
        subprocess.run(args, cwd=yolov5_path, env=yolo_env, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        kili_print("Error while executing YoloV5:")
        for k, v in e.__dict__.items():
            print(k)
            if isinstance(v, bytes):
                print(v.decode("utf-8"))
            else:
                print(v)
        raise AutoMLYoloException()

    shutil.copy(config_data_path, model_output_path)
    df_result = pd.read_csv(os.path.join(model_output_path, "exp", "results.csv"))

    # we take the class loss as the main metric
    return df_result.iloc[-1:][["        val/obj_loss"]].to_numpy()[0][0]  # type: ignore
