import csv
import json
import os
import random
import warnings
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, TypeVar
from warnings import warn

import backoff
import numpy as np
import torch
from kili.client import Kili
from tabulate import tabulate
from typing_extensions import Literal, get_args

from kiliautoml.utils.helper_mock import GENERATE_MOCK, jsonify_mock_data
from kiliautoml.utils.logging import logger
from kiliautoml.utils.memoization import kili_project_memoizer
from kiliautoml.utils.path import AUTOML_CACHE
from kiliautoml.utils.type import (
    AssetFilterArgsT,
    AssetsLazyList,
    AssetStatusT,
    AssetT,
    CategoryIdT,
    CategoryNameT,
    EvalResultsT,
    InputTypeT,
    JobNameT,
    JobsT,
    JobT,
    LabelMergeStrategyT,
    MLTaskT,
    ParityFilterT,
    ProjectIdT,
    ToolT,
)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type:ignore


set_all_seeds(42)


def categories_from_job(job: JobT) -> List[CategoryIdT]:
    """Returns the category id.

    Example:
        - categoryId = "LIGHT_OF_THE_CAR"
        - category name = "light of the car"
    """
    return [cat for cat in job["content"]["categories"].keys()]


def get_content_input_from_job(job: JobT):
    return job["content"].get("input")


def ensure_dir(file_path: str):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


@kili_project_memoizer(sub_dir="get_asset_memoized")
def get_asset_memoized(
    *,
    kili: Kili,
    project_id: ProjectIdT,
    total: Optional[int],
    skip: int,
    status_in: Optional[List[AssetStatusT]] = None,
    asset_filter: Optional[AssetFilterArgsT] = None,
    query_content: bool,
) -> List[Any]:
    kili.assets = backoff.on_exception(backoff.expo, exception=Exception, max_tries=3)(kili.assets)

    additional_filtering_arguments: Dict[str, Any] = {}
    if asset_filter is not None:
        additional_filtering_arguments = {}
        for argument in asset_filter.keys():
            if argument in [
                "asset_id_in",
                "consensus_mark_gt",
                "consensus_mark_lt",
                "external_id_contains",
                "honeypot_mark_gt",
                "honeypot_mark_lt",
                "label_author_in",
                "label_created_at",
                "label_created_at_gt",
                "label_created_at_lt",
                "label_category_search",
                "label_type_in",
                "metadata_where",
                "updated_at_gte",
                "updated_at_lte",
            ]:
                additional_filtering_arguments[argument] = asset_filter[argument]
            else:
                logger.warning(f"You can not filter on this field {argument}")
    fields = [
        "id",
        "externalId",
        "labels.createdAt",
        "labels.jsonResponse",
        "labels.labelType",
    ]
    if query_content:
        fields.append("content")
    assets = kili.assets(
        project_id=project_id,
        first=total,
        skip=skip,
        fields=fields,
        status_in=status_in,
        as_generator=False,
        **additional_filtering_arguments,
    )

    if GENERATE_MOCK:
        jsonify_mock_data(assets, function_name="assets")

    return assets  # type:ignore


def get_assets(
    kili: Kili,
    project_id: ProjectIdT,
    status_in: Optional[List[AssetStatusT]] = None,
    max_assets: Optional[int] = None,
    randomize: bool = False,
    strategy: LabelMergeStrategyT = "last",
    job_name: Optional[JobNameT] = None,
    parity_filter: ParityFilterT = "none",
    asset_filter: Optional[AssetFilterArgsT] = None,
    query_content: bool = True,
) -> AssetsLazyList:
    """
    job_name is used if status_in does not have only unlabeled statuses
    """
    if status_in is not None:
        for status in status_in:
            if status not in get_args(AssetStatusT):
                raise Exception(
                    f"{status} is not a valid asset status. Status must be in"
                    f" {get_args(AssetStatusT)}"
                )
    if status_in is not None:
        logger.info(f"Fetching assets with status in {status_in} from Kili project")
    else:
        logger.info("Fetching assets from Kili project")

    if randomize:
        assets = get_asset_memoized(
            kili=kili,
            project_id=project_id,
            total=None,
            skip=0,
            status_in=status_in,
            asset_filter=asset_filter,
            query_content=query_content,
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
            asset_filter=asset_filter,
            query_content=query_content,
        )

    assets = [AssetT.construct(**asset) for asset in assets]
    assets = AssetsLazyList(assets)
    if status_in is not None:
        only_labeled_status = not any(status in status_in for status in ["TO DO", "ONGOING"])
        if job_name is not None and only_labeled_status:
            assets = filter_labeled_assets(job_name, strategy, assets)

    assets = filter_parity(parity_filter, assets)

    if len(assets) == 0:
        logger.error(f"No {status_in} assets found in project {project_id}.")
        raise Exception("There is no asset matching the query.")
    return assets


def filter_parity(parity_filter: ParityFilterT, assets: AssetsLazyList):
    parity = [0, 1]
    if parity_filter == "keep-even":
        parity = [0]
    if parity_filter == "keep-uneven":
        parity = [1]
    assets_ = [a for a in assets if hash(a.externalId) % 2 in parity]

    return AssetsLazyList(assets_)


TYPE_ORDER = {
    v: i for i, v in enumerate(["REVIEW", "DEFAULT", "PREDICTION", "INFERENCE", "AUTOSAVE"])
}


def _get_label(asset: AssetT, job_name: JobNameT, strategy: LabelMergeStrategyT):
    labels = asset.labels

    # Filter the jobname
    labels = [label for label in labels if job_name in label["jsonResponse"].keys()]

    # XXX: we should probably delete this line
    labels = [label for label in labels if label["labelType"] in ["DEFAULT", "REVIEW"]]

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

    if len(labels) > 0:
        key = first_order if strategy == "first" else last_order
        label = min(labels, key=key)
        return label
    else:
        warn(f"Asset {asset.id} does not have any label available")
        return None


def filter_labeled_assets(
    job_name: JobNameT, strategy: LabelMergeStrategyT, assets: AssetsLazyList
):
    asset_id_to_remove = set()
    for asset in assets:
        label = _get_label(asset, job_name, strategy)
        if label is None:
            asset_id = asset.id
            warnings.warn(f"${asset_id} removed because no labels where available", stacklevel=3)
            asset_id_to_remove.add(asset_id)
        else:
            asset.labels = [label]

    return AssetsLazyList([asset for asset in assets if asset.id not in asset_id_to_remove])


def get_project(kili, project_id: ProjectIdT) -> Tuple[InputTypeT, JobsT, str]:
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


T = TypeVar("T")  # Declare type variable


def set_default(x: Optional[T], x_default: T, x_name: str, x_range: List[T]) -> T:
    if x is None:
        logger.info(f"defaulting to {x_name}={x_default}")
        return x_default
    if x not in x_range:
        logger.warning(f"{x} is not in {x_range}, defaulting to {x_name}={x_default}")
        return x_default
    return x


def get_last_trained_model_path(
    *,
    project_id: ProjectIdT,
    job_name: JobNameT,
    project_path_wildcard: List[str],
    weights_filename: str,
    model_path: Optional[str],
) -> str:
    if model_path is None:
        path_project_models = os.path.join(
            AUTOML_CACHE, project_id, job_name, *project_path_wildcard
        )
        logger.info("Searching models in folder:", path_project_models)
        paths_project_sorted = sorted(glob(path_project_models), reverse=True)
        model_path = None
        while len(paths_project_sorted):
            path_model_candidate = paths_project_sorted.pop(0)
            if len(os.listdir(path_model_candidate)) > 0 and os.path.exists(
                os.path.join(path_model_candidate, weights_filename)
            ):
                model_path = path_model_candidate
                logger.info(f"Trained model found in path: {model_path}")
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
            logger.info("Asset IDs of wrong labels written to: ", json_path)


def not_implemented_job(job_name: JobNameT, ml_task: MLTaskT, tools: List[ToolT]):
    _ = tools
    if "_MARKER" in job_name:
        return
    else:
        logger.error(f"MLTask {ml_task} for job {job_name} is not yet supported")
        logger.error(
            f"""You can use --ignore-job {job_name}
            \n(You can also use the repeatable flag --target-job
            (for example: --target-job job_name1 --target-job job_name2)
            to select one or multiple jobs.)"""
        )
        raise NotImplementedError


def get_mapping_category_name_cat_kili_id(job: JobT):
    cats = job["content"]["categories"]
    mapping_category_name_category_ids: Dict[CategoryNameT, CategoryIdT] = {
        cat["name"]: catId for catId, cat in cats.items()
    }
    return mapping_category_name_category_ids


def create_table(job_name: JobNameT, evaluation: EvalResultsT):
    def get_keys(my_dict):
        keys = list(my_dict.keys())
        keys_int = []
        for key in keys:
            keys_int.extend(list(my_dict[key].keys()))
        return list(set(keys_int))

    # get headers
    keys = get_keys(evaluation)
    keys.sort()
    # get body
    table = []
    table_int = []
    for k, values in evaluation.items():
        table_int.append(k)
        for key in keys:
            if key in values.keys():
                table_int.append(round(values[key], 4))
            else:
                table_int.append("nan")
        table.append(table_int)
        table_int = []
    headers = [job_name] + keys
    return headers, table


def print_and_save_evaluation(
    job_name: JobNameT, evaluation: EvalResultsT, results_folder: Optional[str] = None
):
    headers, table = create_table(job_name, evaluation)
    print(tabulate(table, headers=headers))

    if results_folder is not None:
        if not (os.path.exists(results_folder)):
            os.makedirs(results_folder)

        rows_list = [headers] + table
        with open(
            os.path.join(results_folder, f"{job_name}_results.csv"), "w", newline=""
        ) as eval_file:
            writer = csv.writer(eval_file)
            writer.writerows(rows_list)


def is_contours_detection(input_type, ml_task, content_input, tools):
    return (
        content_input == "radio"
        and input_type == "IMAGE"
        and ml_task == "OBJECT_DETECTION"
        and any(tool in tools for tool in ["semantic", "polygon"])
    )


def curated_job(jobs: JobsT, target_job: List[JobNameT], ignore_job: List[JobNameT]) -> JobsT:
    """Remove from the jobs dict the ignored jobs and keep only the target jobs"""
    assert set(target_job).isdisjoint(ignore_job), "target_job and ignore_job should be disjoint."
    assert set(target_job).issubset(jobs.keys()), f"target_job is not a subset of {jobs.keys()}"
    assert set(ignore_job).issubset(jobs.keys()), f"ignore_job is not a subset of {jobs.keys()}"

    marker_jobs = [job_name for job_name in jobs.keys() if "_MARKER" in job_name]
    ignore_job = list(set(list(ignore_job) + marker_jobs))

    kept_job = list(jobs.keys())
    if target_job:
        kept_job = target_job
    if ignore_job:
        kept_job = list(set(kept_job) - set(ignore_job))

    assert len(kept_job)

    new_job = {}
    for job_name, job in jobs.items():
        if job_name in kept_job:
            new_job[job_name] = job

    return new_job


def dry_run_security(dry_run: bool, entity_to_upload: Literal["predictions", "label errors"]):
    if dry_run is True:
        return dry_run
    logger.info(f"Are you sure You want to send the {entity_to_upload} to Kili? Y/N")
    validation = input()
    if validation in ["N", "n", "No", "NO", "no"]:
        dry_run = True
        logger.info(f"OK, We won't send the {entity_to_upload} to Kili!")
    else:
        logger.info(f"OK, We will send the {entity_to_upload} to Kili!")
    return dry_run
