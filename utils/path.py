import os
from datetime import datetime

from utils.constants import HOME


class Path:
    """
    Paths manager class.
    """

    @staticmethod
    def cache(project_id, sub_dir):
        cache_path = os.path.join(HOME, project_id, sub_dir)
        return cache_path

    # those 3 following functions create the path value.

    @staticmethod
    def model_repository(root_dir, project_id, job_name, model_repository):
        return os.path.join(root_dir, project_id, job_name, model_repository)

    @staticmethod
    def dataset(root_dir, project_id, job_name):
        return os.path.join(root_dir, project_id, job_name, "dataset")

    @staticmethod
    def inference(root_dir, project_id, job_name, model_repository):
        return os.path.join(root_dir, project_id, job_name, model_repository, "inference")

    # those 3 functions use the path value for huggingface.

    @staticmethod
    def append_hf_training_file(job_path):
        return os.path.join(job_path, "dataset", "data.json")

    @staticmethod
    def append_hf_model_folder(job_path, model_framework):
        return os.path.join(
            job_path, "model", str(model_framework), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    @staticmethod
    def append_hf_training_args_folder(model_path):
        return os.path.join(model_path, "training_args")
