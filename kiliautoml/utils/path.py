import os
from datetime import datetime

from kiliautoml.utils.constants import HOME, ModelFrameworkT, ModelRepositoryT


def makedirs_exist_ok(some_function):
    def wrapper(*args, **kwargs):
        res = some_function(*args, **kwargs)
        os.makedirs(res, exist_ok=True)
        return res

    return wrapper


JobPathT = str
ModelRepositoryPathT = str


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
    def cache_memoization(project_id, sub_dir):
        cache_path = os.path.join(HOME, project_id, sub_dir)
        return cache_path

    @staticmethod
    @makedirs_exist_ok
    def job(root_dir, project_id, job_name) -> JobPathT:
        return os.path.join(root_dir, project_id, job_name)

    @staticmethod
    @makedirs_exist_ok
    def model_repository(
        root_dir, project_id, job_name, model_repository: ModelRepositoryT
    ) -> ModelRepositoryPathT:
        return os.path.join(Path.job(root_dir, project_id, job_name), model_repository)


class PathUltralytics:
    @staticmethod
    @makedirs_exist_ok
    def inference(root_dir, project_id, job_name, model_repository: ModelRepositoryT):
        return os.path.join(
            Path.model_repository(root_dir, project_id, job_name, model_repository), "inference"
        )


ModelPathT = str


class PathHF:
    @staticmethod
    @makedirs_exist_ok
    def append_model_folder(
        model_repository_path: ModelRepositoryPathT, model_framework: ModelFrameworkT
    ) -> ModelPathT:
        return os.path.join(
            model_repository_path,
            "model",
            model_framework,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    @staticmethod
    @makedirs_exist_ok
    def append_training_args_folder(model_path: ModelPathT):
        return os.path.join(model_path, "training_args")

    @staticmethod
    @makedirs_exist_ok
    def dataset(model_repository_path: ModelRepositoryPathT):
        return os.path.join(model_repository_path, "dataset")


class PathPytorchVision:
    @staticmethod
    @makedirs_exist_ok
    def append_model_folder(
        model_repository_path: ModelRepositoryPathT,
    ) -> ModelPathT:
        return os.path.join(model_repository_path, "pytorch", "model")

    @staticmethod
    @makedirs_exist_ok
    def append_training_args_folder(model_path: ModelPathT):
        return os.path.join(model_path, "training_args")
