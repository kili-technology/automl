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


def ultralytics_train_yolov5(
    api_key: str,
    path: str,
    job: Dict,
    max_assets: Optional[int],
    json_args: Dict,
    project_id: str,
    model_framework: str,
    label_types: List[str],
    clear_dataset_cache: bool = False,
) -> float:
    yolov5_path = os.path.join(os.getcwd(), "utils", "ultralytics", "yolov5")

    template = env.get_template("kili_template.yml")
    class_names = categories_from_job(job)
    data_path = os.path.join(path, "data")
    config_data_path = os.path.join(yolov5_path, "data", "kili.yaml")
    output_path = os.path.join(
        path, "model", model_framework, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    if clear_dataset_cache and os.path.exists(data_path) and os.path.isdir(data_path):
        kili_print("Dataset cache for this project is being cleared.")
        shutil.rmtree(data_path)

    os.makedirs(output_path)

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

    args_from_json = reduce(lambda x, y: x + y, ([f"--{k}", f"{v}"] for k, v in json_args.items()))
    kili_print("Starting Ultralytics' YoloV5 ...")
    try:
        args = [
            "python",
            "train.py",
            "--data",
            "kili.yaml",
            "--project",
            f"{output_path}",
            *args_from_json,
        ]
        subprocess.run(
            args,
            check=True,
            cwd=f"{yolov5_path}",
        )
    except subprocess.CalledProcessError as e:
        raise AutoMLYoloException("YoloV5 training crashed." + str(e))

    shutil.copy(config_data_path, output_path)
    df_result = pd.read_csv(os.path.join(output_path, "exp", "results.csv"))

    # we take the class loss as the main metric
    return df_result.iloc[-1:][["        val/obj_loss"]].to_numpy()[0][0]
