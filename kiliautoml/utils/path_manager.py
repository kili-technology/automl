import os
from datetime import datetime

from utils.constants import HOME


def get_cache_path(project_id, sub_dir):
    cache_path = os.path.join(HOME, project_id, sub_dir)
    return cache_path


# those 3 following functions create the path value.


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


# those 3 functions use the path value.


def get_huggingface_train_path(path) -> str:
    return os.path.join(path, "dataset", "data.json")


def get_path_model_huggingface(path, model_framework):
    return os.path.join(
        path, "model", str(model_framework), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def get_training_arguments_huggingface(path_model):
    return os.path.join(path_model, "training_args")
