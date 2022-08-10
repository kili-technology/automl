import os
import shutil
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from kiliautoml.utils.type import (
    JobNameT,
    MLBackendT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
)

AUTOML_CACHE = os.getenv(
    "KILIAUTOML_CACHE", os.path.join(os.getenv("HOME"), ".cache", "kili", "automl")  # type:ignore
)


TFunc = Callable[..., Any]


def makedirs_exist_ok(function: TFunc) -> TFunc:
    @wraps(function)
    def wrapper(*args, **kwargs):
        res = function(*args, **kwargs)
        os.makedirs(res, exist_ok=True)
        return res

    return wrapper


def reset_dir(function: TFunc) -> TFunc:
    @wraps(function)
    def wrapper(*args, **kwargs):
        res = function(*args, **kwargs)
        shutil.rmtree(res, ignore_errors=True)
        os.makedirs(res, exist_ok=True)
        return res

    return wrapper


JobDirT = str
ModelRepositoryDirT = str
ModelDirT = str
ModelPathT = str


class Path:
    """
    Paths manager class.

    A project is composed of the following nested directories:


    ├── cl0wihlop3rwc0mtj9np28ti2 # project_id
    │   └── DETECTION # job_name
    │       └── ultralytics # model_repository
    │           ├── inference
    │           └── model
    │               └── pytorch
    └── joblib
        ├── kiliautoml
        │   └── utils
        │       └── helpers
        │           └── download_image


    """

    @staticmethod
    @makedirs_exist_ok
    def cache_memoization_dir(project_id, sub_dir):
        cache_path = os.path.join(AUTOML_CACHE, project_id, sub_dir)
        return cache_path

    @staticmethod
    @makedirs_exist_ok
    def job_dir(project_id, job_name: JobNameT) -> JobDirT:
        return os.path.join(AUTOML_CACHE, project_id, job_name)

    @staticmethod
    @makedirs_exist_ok
    def model_repository_dir(
        project_id: ProjectIdT, job_name: JobNameT, model_repository: ModelRepositoryT
    ) -> ModelRepositoryDirT:
        return os.path.join(Path.job_dir(project_id, job_name), model_repository)


"""
Once we have the model repository dir, we can create the following nested directories:
"""


class PathUltralytics:
    @staticmethod
    @makedirs_exist_ok
    def inference_dir(project_id, job_name: JobNameT, model_repository: ModelRepositoryT):
        return os.path.join(
            Path.model_repository_dir(project_id, job_name, model_repository),
            "inference",
        )

    ULTRALYTICS_REL_PATH = os.path.join("kiliautoml", "utils", "ultralytics")
    YOLOV5_REL_PATH = os.path.join(ULTRALYTICS_REL_PATH, "yolov5")


class PathHF:
    @staticmethod
    @makedirs_exist_ok
    def append_model_folder(
        model_repository_dir: ModelRepositoryDirT, ml_backend: MLBackendT
    ) -> ModelDirT:
        return os.path.join(
            model_repository_dir,
            "model",
            ml_backend,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    @staticmethod
    @makedirs_exist_ok
    def append_training_args_dir(model_dir: ModelDirT):
        return os.path.join(model_dir, "training_args")

    @staticmethod
    @makedirs_exist_ok
    def dataset_dir(model_repository_dir: ModelRepositoryDirT):
        return os.path.join(model_repository_dir, "dataset")


class PathPytorchVision:
    @staticmethod
    @makedirs_exist_ok
    def append_model_dir(
        model_repository_dir: ModelRepositoryDirT,
    ) -> ModelDirT:
        return os.path.join(model_repository_dir, "pytorch", "model")

    @staticmethod
    def append_model_path(model_dir: ModelDirT, model_name: ModelNameT) -> ModelPathT:
        return os.path.join(model_dir, model_name)

    @staticmethod
    @makedirs_exist_ok
    def append_data_dir(model_dir: ModelDirT) -> ModelPathT:
        return os.path.join(model_dir, "data")

    @staticmethod
    @makedirs_exist_ok
    def append_training_args_folder(model_dir: ModelDirT):
        return os.path.join(model_dir, "training_args")


# TODO: Use @reset_dir forthe other classes, not just PathDetectron2


class PathDetectron2:
    @staticmethod
    @makedirs_exist_ok
    def append_model_dir(
        model_repository_dir: ModelRepositoryDirT,
    ) -> ModelDirT:
        return os.path.join(model_repository_dir, "pytorch", "model")

    @staticmethod
    @reset_dir
    def append_data_dir(model_repository_dir: ModelRepositoryDirT):
        return os.path.join(model_repository_dir, "data")

    @staticmethod
    @reset_dir
    def append_output_evaluation(model_repository_dir: ModelRepositoryDirT):
        return os.path.join(model_repository_dir, "evaluation")

    @staticmethod
    @reset_dir
    def append_output_visualization(model_repository_dir: ModelRepositoryDirT):
        return os.path.join(model_repository_dir, "prediction_visualization")
