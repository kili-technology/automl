import os
import subprocess
from typing import Dict, List, Optional
from datetime import datetime
import shutil
from functools import reduce

from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd

from utils.helpers import categories_from_job, kili_print

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
    max_assets: Optional[int],
    json_args: Dict,
    project_id: str,
    model_framework: str,
    label_types: List[str],
    title: str,
    clear_dataset_cache: bool = False,
) -> float:
    yolov5_path = os.path.join(os.getcwd(), "utils", "ultralytics", "yolov5")

    template = env.get_template("kili_template.yml")
    class_names = categories_from_job(job)
    data_path = os.path.join(path, "data")
    config_data_path = os.path.join(yolov5_path, "data", "kili.yaml")

    if clear_dataset_cache and os.path.exists(data_path) and os.path.isdir(data_path):
        kili_print("Dataset cache for this project is being cleared.")
        shutil.rmtree(data_path)

    model_output_path = get_output_path_bbox(title, path, model_framework)
    os.makedirs(model_output_path, exist_ok=True)

    with open(config_data_path, "w") as f:
        f.write(
            template.render(
                data_path=data_path,
                class_names=class_names,
                number_classes=len(class_names),
                kili_api_key=api_key,
                project_id=project_id,
                label_types=label_types,
                max_assets=max_assets,
            )
        )

    if not json_args:
        json_args = {"epochs": 50}
        kili_print("No arguments were passed to the train function. Defaulting to epochs=50.")
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
