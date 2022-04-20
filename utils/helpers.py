import os
import random
from io import BytesIO
import shutil
from typing import Any, List, Optional, Dict, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from glob import glob
from joblib import Memory
from tqdm import tqdm
from PIL import Image
from PIL.Image import Image as PILImage
import requests
from utils.active_learning_demo import select_assets_training_active_learning_cycle

from utils.constants import HOME
from utils.helpers_functools import kili_print


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type:ignore


set_all_seeds(42)


def kili_project_memoizer(
    sub_dir: str,
):
    """Decorator factory for memoizing a function that takes a project_id as input."""

    def decorator(some_function):
        def wrapper(*args, **kwargs):
            project_id = kwargs.get("project_id")
            if not project_id:
                raise ValueError("project_id not specified in a keyword argument")
            cache_path = os.path.join(HOME, project_id, sub_dir)
            memory = Memory(cache_path, verbose=0)
            return memory.cache(some_function)(*args, **kwargs)

        return wrapper

    return decorator


def kili_memoizer(some_function):
    def wrapper(*args, **kwargs):
        memory = Memory(HOME, verbose=0)
        return memory.cache(some_function)(*args, **kwargs)

    return wrapper


def categories_from_job(job: Dict):
    return list(job["content"]["categories"].keys())


def ensure_dir(file_path: str):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


class JobPredictions:
    def __init__(
        self,
        job_name: str,
        external_id_array: List[str],
        json_response_array: List[Any],
        model_name_array: List[str],
        predictions_probability: List[float],
    ):
        self.job_name = job_name
        self.external_id_array = external_id_array
        self.json_response_array = json_response_array
        self.model_name_array = model_name_array
        self.predictions_probability = predictions_probability

        n_assets = len(external_id_array)

        # assert all lists are compatible
        same_len = n_assets == len(json_response_array)
        assert same_len, "external_id_array and json_response_array must have the same length"

        same_len = n_assets == len(model_name_array)
        assert same_len, "external_id_array and model_name_array must have the same length"

        same_len = n_assets == len(predictions_probability)
        assert same_len, "external_id_array and predictions_probability must have the same length"

        # assert no duplicates
        assert (
            len(set(external_id_array)) == n_assets
        ), "external_id_array must not contain duplicates"

        kili_print(f"JobPredictions: {n_assets} assets successfully created for job {job_name}.")

    def __repr__(self):
        return f"JobPredictions(job_name={self.job_name}, nb_assets={len(self.external_id_array)})"


@kili_project_memoizer(sub_dir="get_asset_memoized")
def _get_asset_memoized(*, kili, project_id, first, skip):
    return kili.assets(
        project_id=project_id,
        first=first,
        skip=skip,
        disable_tqdm=True,
        fields=[
            "id",
            "externalId",
            "content",
            "labels.createdAt",
            "labels.jsonResponse",
            "labels.labelType",
        ],
    )


def asset_is_kept(asset, labeling_statuses: List[str] = ["LABELED", "UNLABELED"]) -> bool:
    labeled = len(asset["labels"]) > 0
    unlabeled = len(asset["labels"]) == 0
    if "LABELED" in labeling_statuses:
        return labeled
    if "UNLABELED" in labeling_statuses:
        return unlabeled
    return False


def get_assets(
    *,
    kili,
    project_id: str,
    active_learning_demo: bool,
    label_types: List[str] = ["DEFAULT", "REVIEW"],
    max_assets: Optional[int] = None,
    labeling_statuses: List[str] = ["LABELED", "UNLABELED"],
    active_learning_demo: bool = False,
) -> List[Dict]:
    if active_learning_demo:
        assert labeling_statuses == [
            "LABELED"
        ], "active_learning_demo is only supported for LABELED assets."
        assert max_assets is None, "max_assets must be None for active_learning_demo."

    total = kili.count_assets(project_id=project_id)

    first = min(100, total)
    assets = []
    for skip in tqdm(range(0, total, first)):
        assets += _get_asset_memoized(kili=kili, project_id=project_id, first=first, skip=skip)
    assets = [
        {
            **a,
            "labels": [
                line
                for line in sorted(a["labels"], key=lambda l: l["createdAt"])
                if line["labelType"] in label_types
            ][-1:],
        }
        for a in assets
    ]
    if not labeling_statuses:
        raise ValueError("labeling_statuses must be a non-empty list.")
    assets = [a for a in assets if asset_is_kept(a, labeling_statuses)]

    if active_learning_demo:
        assets = select_assets_training_active_learning_cycle(project_id, assets)

    max_assets = min(max_assets, len(assets)) if max_assets else len(assets)
    assets = assets[:max_assets]
    return assets


def get_project(kili, project_id: str) -> Tuple[str, Dict, str]:
    projects = kili.projects(project_id=project_id, fields=["inputType", "jsonInterface", "title"])
    if len(projects) == 0:
        raise ValueError("no such project")
    input_type = projects[0]["inputType"]
    jobs = projects[0]["jsonInterface"].get("jobs", {})
    title = projects[0]["title"]
    return input_type, jobs, title


def build_model_repository_path(
    root_dir: str, project_id: str, job_name: str, model_repository: str
) -> str:
    return os.path.join(root_dir, project_id, job_name, model_repository)


def build_dataset_path(root_dir: str, project_id: str, job_name: str) -> str:
    return os.path.join(root_dir, project_id, job_name, "dataset")


def build_inference_path(
    root_dir: str, project_id: str, job_name: str, model_repository: str
) -> str:
    return os.path.join(root_dir, project_id, job_name, model_repository, "inference")


def parse_label_types(label_types: Optional[str]) -> List[str]:
    return label_types.split(",") if label_types else ["DEFAULT", "REVIEW"]


def set_default(x: str, x_default: str, x_name: str, x_range: List[str]) -> str:
    if x not in x_range:
        kili_print(f"defaulting to {x_name}={x_default}")
        x = x_default
    return x


def get_last_trained_model_path(
    *,
    project_id: str,
    job_name: str,
    project_path_wildcard: List[str],
    weights_filename: str,
    model_path: Optional[str],
) -> str:
    if model_path is None:
        path_project_models = os.path.join(HOME, project_id, job_name, *project_path_wildcard)
        kili_print("searching models in folder:", path_project_models)
        paths_project_sorted = sorted(glob(path_project_models), reverse=True)
        model_path = None
        while len(paths_project_sorted):
            path_model_candidate = paths_project_sorted.pop(0)
            if len(os.listdir(path_model_candidate)) > 0 and os.path.exists(
                os.path.join(path_model_candidate, weights_filename)
            ):
                model_path = path_model_candidate
                kili_print(f"Trained model found in path: {model_path}")
                break
        if model_path is None:
            raise Exception("No trained model found for job {job}. Exiting ...")
    return model_path


@dataclass
class DownloadedImages:
    id: str
    externalId: str
    filename: str
    image: PILImage


@kili_memoizer
def download_image(api_key, asset_content):
    img_data = requests.get(
        asset_content,
        headers={
            "Authorization": f"X-API-Key: {api_key}",
        },
    ).content

    image = Image.open(BytesIO(img_data))
    return image


def download_project_images(
    api_key,
    assets,
    inference_path: Optional[str] = None,
) -> List[DownloadedImages]:
    kili_print("Downloading project images...")
    downloaded_images = []
    for asset in tqdm(assets):
        image = download_image(api_key, asset["content"])
        format = str(image.format or "")

        filename = ""
        if inference_path:
            filename = os.path.join(inference_path, asset["id"] + "." + format.lower())

            with open(filename, "w") as fp:
                image.save(fp, format)  # type: ignore

        downloaded_images.append(
            DownloadedImages(
                id=asset["id"],
                externalId=asset["externalId"],
                filename=filename or "",
                image=image,
            )
        )
    return downloaded_images


def clear_automl_cache():
    if os.path.exists(HOME):
        shutil.rmtree(HOME)
