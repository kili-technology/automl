import os
import shutil
from functools import wraps
from typing import Any, Callable, List, Optional

from joblib import Memory
from typing_extensions import get_args

from kiliautoml.utils.path import AUTOML_CACHE, ModelRepositoryDirT, Path, PathHF
from kiliautoml.utils.type import (
    CommandT,
    JobNameT,
    MLBackendT,
    ModelRepositoryT,
    ProjectIdT,
)

TFunc = Callable[..., Any]


def kili_project_memoizer(
    sub_dir: str,
):
    """Decorator factory for memoizing a function that takes a project_id as input."""

    def decorator(some_function: TFunc) -> TFunc:
        @wraps(some_function)
        def wrapper(*args, **kwargs):
            project_id = kwargs.get("project_id")
            if not project_id:
                raise ValueError("project_id not specified in a keyword argument")
            cache_path = Path.cache_memoization_dir(project_id, sub_dir)
            print("cache_path", cache_path)
            memory = Memory(cache_path, verbose=1)
            return memory.cache(some_function, ignore=["kili"])(*args, **kwargs)

        return wrapper

    return decorator


def kili_memoizer(some_function: TFunc) -> TFunc:
    """We ignore the argument asset_content"""

    @wraps(some_function)
    def wrapper(*args, **kwargs):
        memory = Memory(AUTOML_CACHE, verbose=0)
        return memory.cache(some_function, ignore=["asset_content"])(*args, **kwargs)

    return wrapper


def clear_command_cache(
    command: CommandT,
    project_id: ProjectIdT,
    job_name: JobNameT,
    model_repository: Optional[ModelRepositoryT] = None,
    ml_backend: Optional[MLBackendT] = None,
):
    """If model_repository is None, then it clears for every modelRepository cache."""
    sub_dirs = ["get_asset_memoized"]
    cache_paths: List[ModelRepositoryDirT] = [
        Path.cache_memoization_dir(project_id, sub_dir) for sub_dir in sub_dirs
    ]

    if model_repository is None:
        model_repositories: List[ModelRepositoryT] = get_args(ModelRepositoryT)  # type: ignore
    else:
        model_repositories = [model_repository]

    for model_repository in model_repositories:
        if command in ["train", "label_errors"]:
            assert job_name is not None
            assert model_repository is not None
            path = Path.model_repository_dir(
                project_id=project_id,
                job_name=job_name,
                model_repository=model_repository,
            )
            if ml_backend is None:
                cache_paths.append(path)
            else:
                cache_paths.append(PathHF.append_model_folder(path, ml_backend))

        for cache_path in cache_paths:
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
