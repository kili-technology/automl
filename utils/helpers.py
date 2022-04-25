import os
import random
from io import BytesIO
import json
import shutil
from typing import Any, List, Optional, Dict, Tuple

import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
from dataclasses import dataclass

import torch
import numpy as np
from glob import glob
from termcolor import colored
from joblib import Memory
from tqdm import tqdm
from PIL import Image
from PIL.Image import Image as PILImage
import requests

from utils.constants import HOME


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

        kili_print(
            f"JobPredictions: {n_assets} predictions successfully created for job {job_name}."
        )

    def __repr__(self):
        return f"JobPredictions(job_name={self.job_name}, nb_assets={len(self.external_id_array)})"


labeling_statusT = Literal["LABELED", "UNLABELED"]
status_inT = Literal["TODO", "ONGOING", "LABELED", "REVIEWED", "DEFAULT"]
label_typeT = Literal["DEFAULT", "REVIEW"]


@kili_project_memoizer(sub_dir="get_asset_memoized")
def get_asset_memoized(
    *,
    kili,
    project_id,
    first,
    skip,
    status_in: Optional[List[status_inT]] = None,
    label_type_in: Optional[List[label_typeT]] = None,
) -> List[Dict]:
    return kili.assets(
        project_id=project_id,
        first=first,
        skip=skip,
        fields=[
            "id",
            "externalId",
            "content",
            "labels.createdAt",
            "labels.jsonResponse",
            "labels.labelType",
        ],
        status_in=status_in,
        label_type_in=label_type_in,
        as_generator=False,
    )


def asset_is_kept(
    asset, labeling_statuses: List[labeling_statusT] = ["LABELED", "UNLABELED"]
) -> bool:
    labeled = len(asset["labels"]) > 0
    unlabeled = len(asset["labels"]) == 0
    return ("LABELED" in labeling_statuses and labeled) or (
        "UNLABELED" in labeling_statuses and unlabeled
    )


def compute_status_in(labeling_statuses: List[labeling_statusT]) -> List[status_inT]:
    labeled: List[status_inT] = ["LABELED", "REVIEWED"]
    unlabeled: List[status_inT] = ["TODO", "ONGOING"]
    status_in = []
    if "LABELED" in labeling_statuses:
        status_in += labeled
    if "UNLABELED" in labeling_statuses:
        status_in += unlabeled
    return status_in


def get_assets(
    kili,
    project_id: str,
    label_type_in: List[label_typeT] = ["DEFAULT", "REVIEW"],
    max_assets: Optional[int] = None,
    labeling_statuses: List[labeling_statusT] = ["LABELED", "UNLABELED"],
) -> List[Dict]:
    if not labeling_statuses:
        raise ValueError("labeling_statuses must be a non-empty list.")

    total = kili.count_assets(project_id=project_id)
    total = total if max_assets is None else min(total, max_assets)

    status_in = compute_status_in(labeling_statuses)
    assets = get_asset_memoized(
        kili=kili,
        project_id=project_id,
        first=total,
        skip=0,
        status_in=status_in,
        label_type_in=label_type_in,
    )

    if len(assets) == 0:
        raise Exception("There is no asset matching the query. Exiting...")
    return assets


def get_project(kili, project_id: str) -> Tuple[str, Dict, str]:
    projects = kili.projects(project_id=project_id, fields=["inputType", "jsonInterface", "title"])
    if len(projects) == 0:
        raise ValueError("no such project")
    input_type = projects[0]["inputType"]
    jobs = projects[0]["jsonInterface"].get("jobs", {})
    title = projects[0]["title"]
    return input_type, jobs, title


def kili_print(*args, **kwargs) -> None:
    print(colored("kili:", "yellow", attrs=["bold"]), *args, **kwargs)


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


def save_errors(found_errors, job_path):
    found_errors_dict = {"assetIds": found_errors}
    found_errors_json = json.dumps(found_errors_dict, sort_keys=True, indent=4)
    if found_errors_json is not None:
        json_path = os.path.join(job_path, "error_labels.json")
        with open(json_path, "wb") as output_file:
            output_file.write(found_errors_json.encode("utf-8"))
            kili_print("Asset IDs of wrong labels written to: ", json_path)


def upload_errors_to_kili(found_errors, kili):
    kili_print("Updating metadatas for the concerned assets")
    first = min(100, len(found_errors))
    for skip in tqdm(range(0, len(found_errors), first)):
        error_assets = kili.assets(
            asset_id_in=found_errors[skip : skip + first], fields=["id", "metadata"]
        )
        asset_ids = [asset["id"] for asset in error_assets]
        new_metadatas = [asset["metadata"] for asset in error_assets]

        for meta in new_metadatas:
            meta["labeling_error"] = True

        kili.update_properties_in_assets(asset_ids=asset_ids, json_metadatas=new_metadatas)
