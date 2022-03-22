import os
from typing import List, Optional, Dict, Tuple
from glob import glob
from numpy import void

from termcolor import colored
from joblib import Memory

from tqdm import tqdm

from utils.constants import HOME

memory = Memory(".cachedir")

def categories_from_job(job: Dict):
    return list(job["content"]["categories"].keys())


def ensure_dir(file_path: str):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


@memory.cache
def get_assets(kili, project_id: str, label_types: List[str], max_assets: Optional[int] = None) -> List[Dict]:
    total = kili.count_assets(project_id=project_id)
    total = total if max_assets is None else min(total, max_assets)

    first = min(100, total)
    assets = []
    for skip in tqdm(range(0, total, first)):
        assets += kili.assets(
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
    assets = [
        {
            **a,
            "labels": [
                l
                for l in sorted(a["labels"], key=lambda l: l["createdAt"])
                if l["labelType"] in label_types
            ][-1:],
        }
        for a in assets
    ]
    assets = [a for a in assets if len(a["labels"]) > 0]
    return assets


def get_project(kili, project_id: str) -> Tuple[str, Dict]:
    projects = kili.projects(
        project_id=project_id, fields=["inputType", "jsonInterface"]
    )
    if len(projects) == 0:
        raise ValueError("no such project")
    input_type = projects[0]["inputType"]
    jobs = projects[0]["jsonInterface"].get("jobs", {})
    return input_type, jobs


def kili_print(*args, **kwargs) -> void:
    print(colored("kili:", "yellow", attrs=["bold"]), *args, **kwargs)


def build_model_repository_path(
    root_dir: str, project_id: str, job_name: str, model_repository: str
) -> str:
    return os.path.join(root_dir, project_id, job_name, model_repository)


def build_dataset_path(
    root_dir: str, project_id: str, job_name: str
) -> str:
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


def get_last_trained_model_path(job_name: str, project_id: str, model_path: str, project_path_wildcard: List[str], weights_filename: str) -> str:
    if model_path is None:
        path_project_models = os.path.join(
            HOME, project_id, job_name, *project_path_wildcard
        )
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
