import os
import shutil
from typing import List, Optional

from joblib import Memory
from typing_extensions import get_args

from kiliautoml.utils.constants import HOME, ModelFrameworkT, ModelRepositoryT
from kiliautoml.utils.path import Path, PathHF
from kiliautoml.utils.type import CommandT


def kili_project_memoizer(
    sub_dir: str,
):
    """Decorator factory for memoizing a function that takes a project_id as input."""

    def decorator(some_function):
        def wrapper(*args, **kwargs):
            project_id = kwargs.get("project_id")
            if not project_id:
                raise ValueError("project_id not specified in a keyword argument")
            cache_path = Path.cache_memoization_dir(project_id, sub_dir)
            memory = Memory(cache_path, verbose=0)
            return memory.cache(some_function)(*args, **kwargs)

        return wrapper

    return decorator


def kili_memoizer(some_function):
    def wrapper(*args, **kwargs):
        memory = Memory(HOME, verbose=0)
        return memory.cache(some_function)(*args, **kwargs)

    return wrapper


def clear_automl_cache(
    command: CommandT,
    project_id: str,
    job_name: str,
    model_repository: Optional[ModelRepositoryT] = None,
    model_framework: Optional[ModelFrameworkT] = None,
):
    """If model_repository is None, then it clears for every modelRepository cache."""
    sub_dirs = ["get_asset_memoized"]
    cache_paths = [Path.cache_memoization_dir(project_id, sub_dir) for sub_dir in sub_dirs]

    if model_repository is None:
        model_repositories: List[ModelRepositoryT] = get_args(ModelRepositoryT)  # type: ignore
    else:
        model_repositories = [model_repository]

    for model_repository in model_repositories:
        if command in ["train", "label_errors"]:
            assert job_name is not None
            assert model_repository is not None
            path = Path.model_repository_dir(
                root_dir=HOME,
                project_id=project_id,
                job_name=job_name,
                model_repository=model_repository,
            )
            if model_framework is None:
                cache_paths.append(path)
            else:
                cache_paths.append(PathHF.append_model_folder(path, model_framework))

        for cache_path in cache_paths:
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
