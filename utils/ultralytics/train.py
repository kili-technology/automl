import os
import subprocess
from typing import Dict
from datetime import datetime
import shutil
from functools import reduce
import math
import re
from tqdm.auto import tqdm

from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd

from utils.helpers import categories_from_job, download_image, kili_print

env = Environment(
    loader=FileSystemLoader(os.path.abspath("utils/ultralytics")),
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


def ultralytics_train_yolov5(
    *,
    api_key: str,
    path: str,
    job: Dict,
    assets,
    json_args: Dict,
    model_framework: str,
    title: str,
    clear_dataset_cache: bool = False,
) -> float:
    yolov5_path = os.path.join(os.getcwd(), "utils", "ultralytics", "yolov5")

    class_names = categories_from_job(job)
    data_path = os.path.join(path, "data")
    config_data_path = os.path.join(yolov5_path, "data", "kili.yaml")

    if clear_dataset_cache and os.path.exists(data_path) and os.path.isdir(data_path):
        kili_print("Dataset cache for this project is being cleared.")
        shutil.rmtree(data_path)

    model_output_path = get_output_path_bbox(title, path, model_framework)
    os.makedirs(model_output_path, exist_ok=True)

    template = env.get_template("kili_template.yml")
    with open(config_data_path, "w") as f:
        f.write(
            template.render(
                data_path=data_path,
                class_names=class_names,
                number_classes=len(class_names),
                kili_api_key=api_key,
            )
        )

    if not json_args:
        json_args = {"epochs": 50}
        kili_print("No arguments were passed to the train function. Defaulting to epochs=50.")
    print("Downloading datasets from Kili")
    train_val_proportions = [0.8, 0.1]
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
        path_split = os.path.join(data_path, "images", name_split)
        print(f"Building {name_split} in {path_split} ...")
        os.makedirs(path_split, exist_ok=True)
        for asset in tqdm(assets_split, desc=f"Downloading {name_split}", unit="assets"):
            img_data = download_image(api_key, asset["content"])
            with open(os.path.join(path_split, asset["id"] + ".jpg"), "wb") as handler:
                handler.write(img_data.tobytes())

        names = class_names
        path_labels = re.sub("/images/", "/labels/", path_split)
        print(path_labels)
        os.makedirs(path_labels, exist_ok=True)
        for asset in assets_split:
            with open(os.path.join(path_labels, asset["id"] + ".txt"), "w") as handler:
                json_response = asset["labels"][0]["jsonResponse"]
                for job in json_response.values():
                    write_to_yolo_format(job, handler, names)

    args_from_json = reduce(lambda x, y: x + y, ([f"--{k}", f"{v}"] for k, v in json_args.items()))
    kili_print("Starting Ultralytics' YoloV5 ...")
    try:
        yolo_env = os.environ.copy()
        yolo_env["WANDB_DISABLED"] = "false"
        args = [
            "python",
            "train.py",
            "--data",
            "kili.yaml",
            "--project",
            f"{model_output_path}",
            "--upload_dataset",  # wandb
            *args_from_json,
        ]
        subprocess.run(args, check=True, cwd=f"{yolov5_path}", env=yolo_env)
    except subprocess.CalledProcessError as e:
        raise AutoMLYoloException("YoloV5 training crashed." + str(e))

    shutil.copy(config_data_path, model_output_path)
    df_result = pd.read_csv(os.path.join(model_output_path, "exp", "results.csv"))

    # we take the class loss as the main metric
    return df_result.iloc[-1:][["        val/obj_loss"]].to_numpy()[0][0]  # type: ignore


def write_to_yolo_format(job, handler, names):
    for annotation in job.get("annotations", []):
        name = annotation["categories"][0]["name"]
        try:
            category = names.index(name)
        except ValueError:
            raise ValueError(f"{name} not in {names}")
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
