import json
import os
import random
import warnings
from datetime import datetime
from glob import glob
from typing import Any, List, Optional, Tuple
from warnings import warn

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm
from typing_extensions import get_args

from kiliautoml.utils.constants import HOME, InputTypeT, MLTaskT
from kiliautoml.utils.helper_mock import GENERATE_MOCK, jsonify_mock_data
from kiliautoml.utils.memoization import kili_project_memoizer
from kiliautoml.utils.type import AssetStatusT, AssetT, JobsT, JobT, LabelMergeStrategyT


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type:ignore


set_all_seeds(42)

TYPE_ORDER = {
    v: i for i, v in enumerate(["REVIEW", "DEFAULT", "PREDICTION", "INFERENCE", "AUTOSAVE"])
}


def last_order(json_response):
    return (
        TYPE_ORDER[json_response["labelType"]],
        -datetime.strptime(json_response["createdAt"], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp(),
    )


def first_order(json_response):
    return (
        TYPE_ORDER[json_response["labelType"]],
        datetime.strptime(json_response["createdAt"], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp(),
    )


def categories_from_job(job: JobT):
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


@kili_project_memoizer(sub_dir="get_asset_memoized")
def get_asset_memoized(
    *,
    kili,
    project_id,
    total,
    skip,
    status_in: Optional[List[AssetStatusT]] = None,
) -> List[AssetT]:
    assets = kili.assets(
        project_id=project_id,
        first=total,
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
        as_generator=False,
    )

    if GENERATE_MOCK:
        jsonify_mock_data(assets, function_name="assets")
    return assets


def get_assets(
    kili,
    project_id: str,
    status_in: Optional[List[AssetStatusT]] = None,
    max_assets: Optional[int] = None,
    randomize: bool = False,
) -> List[AssetT]:

    if status_in is not None:
        for status in status_in:
            if status not in get_args(AssetStatusT):
                raise Exception(
                    f"{status} is not a valid asset status. Status must be in"
                    f" {get_args(AssetStatusT)}"
                )
    if status_in is not None:
        kili_print(f"Downloading assets with status in {status_in} from Kili project")
    else:
        kili_print("Downloading assets from Kili project")

    if randomize:
        assets = get_asset_memoized(
            kili=kili,
            project_id=project_id,
            total=None,
            skip=0,
            status_in=status_in,
        )
        random.shuffle(assets)
        assets = assets[:max_assets]

    else:
        assets = get_asset_memoized(
            kili=kili,
            project_id=project_id,
            total=max_assets,
            skip=0,
            status_in=status_in,
        )

    if len(assets) == 0:
        kili_print(f"No {status_in} assets found in project {project_id}.")
        raise Exception("There is no asset matching the query.")
    return assets


def get_label(
    asset: AssetT, job_name: str, ml_task: Optional[MLTaskT], strategy: LabelMergeStrategyT
):
    labels = asset["labels"]
    # for CLASSIFICATION task, we can only accept label with
    if ml_task == "CLASSIFICATION":
        labels = [label for label in labels if job_name in label["jsonResponse"].keys()]
    if len(labels) > 0:
        key = first_order if strategy == "first" else last_order
        return min(labels, key=key)
    else:
        warn(f"Asset {asset['id']} does not have any label available")
        return None


def get_labeled_assets(
    kili,
    project_id: str,
    job_name: str,
    ml_task: Optional[MLTaskT],
    status_in: Optional[List[AssetStatusT]] = None,
    max_assets: Optional[int] = None,
    randomize: bool = False,
    strategy: LabelMergeStrategyT = "last",
) -> List[AssetT]:
    assets = get_assets(kili, project_id, status_in, max_assets=max_assets, randomize=randomize)
    asset_id_to_remove = set()
    for asset in assets:
        label = get_label(asset, job_name, ml_task, strategy)
        if label is None:
            asset_id = asset["id"]
            warnings.warn(f"${asset_id} removed because no labels where available")
            asset_id_to_remove.add(asset_id)
        else:
            asset["labels"] = [label]

    return [asset for asset in assets if asset["id"] not in asset_id_to_remove]


def get_project(kili, project_id: str) -> Tuple[InputTypeT, JobsT, str]:
    projects = kili.projects(project_id=project_id, fields=["inputType", "jsonInterface", "title"])
    if GENERATE_MOCK:
        jsonify_mock_data(projects, function_name="projects")
    if len(projects) == 0:
        raise ValueError(
            "no such project. Maybe your KILI_API_KEY does not belong to a member of the project."
        )
    input_type = projects[0]["inputType"]
    jobs = projects[0]["jsonInterface"].get("jobs", {})
    title = projects[0]["title"]
    return input_type, jobs, title


def kili_print(*args, **kwargs) -> None:
    print(colored("kili:", "yellow", attrs=["bold"]), *args, **kwargs)


def set_default(x, x_default, x_name: str, x_range: List):  # type: ignore
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
        kili_print("Searching models in folder:", path_project_models)
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
            raise Exception(f"No trained model found for job {job_name}. Exiting ...")
    return model_path


def save_errors(found_errors, job_path: str):
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
    for skip in tqdm(
        range(0, len(found_errors), first), desc="Updating asset metadata with labeling error flag"
    ):
        error_assets = kili.assets(
            asset_id_in=found_errors[skip : skip + first], fields=["id", "metadata"]
        )
        asset_ids = [asset["id"] for asset in error_assets]
        new_metadatas = [asset["metadata"] for asset in error_assets]

        for meta in new_metadatas:
            meta["labeling_error"] = True

        kili.update_properties_in_assets(asset_ids=asset_ids, json_metadatas=new_metadatas)


def not_implemented_job(job_name, ml_task):
    kili_print(f"MLTask {ml_task} for job {job_name} is not yet supported")
    kili_print(
        "You can use the repeatable flag --target-job "
        "(for example: --target-job job_name1 --target-job job_name2) "
        "to select one or multiple jobs."
    )
    raise NotImplementedError
