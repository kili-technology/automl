from abc import abstractmethod
from typing import List, Optional

from typing_extensions import TypedDict

from kiliautoml.utils.helper_label_error import ErrorRecap
from kiliautoml.utils.helpers import set_default
from kiliautoml.utils.path import Path
from kiliautoml.utils.type import (
    AssetsLazyList,
    DictTrainingInfosT,
    JobNameT,
    JobPredictions,
    JobT,
    MLBackendT,
    MLTaskT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
)


class BaseInitArgs(TypedDict):
    job: JobT
    job_name: JobNameT
    model_name: Optional[ModelNameT]
    project_id: ProjectIdT


class BaseTrainArgs(TypedDict):
    assets: AssetsLazyList
    epochs: int
    batch_size: int
    clear_dataset_cache: bool
    disable_wandb: bool
    verbose: int


class BaseModel:
    ml_task: MLTaskT  # type: ignore
    model_repository: ModelRepositoryT  # type: ignore
    ml_backend: MLBackendT  # type: ignore
    advised_model_names: List[ModelNameT]  # type: ignore

    def __init__(
        self,
        *,
        job: JobT,
        job_name: JobNameT,
        project_id: ProjectIdT,
        model_name: Optional[ModelNameT],
        advised_model_names: List[ModelNameT],
    ) -> None:
        self.job = job
        self.job_name = job_name
        self.model_name: ModelNameT = model_name or self.advised_model_names[0]
        self.project_id = project_id

        self.model_repository_dir = Path.model_repository_dir(
            project_id, job_name, self.model_repository
        )

        self.model_name = set_default(
            model_name, advised_model_names[0], "model_name", advised_model_names
        )

    @abstractmethod
    def train(
        self,
        *,
        assets: AssetsLazyList,
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
        disable_wandb: bool,
        verbose: int,
        **kwargs,
    ) -> DictTrainingInfosT:
        pass

    @abstractmethod
    def predict(
        self,
        *,
        assets: AssetsLazyList,
        model_path: Optional[str],
        from_project: Optional[ProjectIdT],
        batch_size: int,
        verbose: int,
        clear_dataset_cache: bool,
    ) -> JobPredictions:
        pass

    @abstractmethod
    def find_errors(
        self,
        *,
        assets: AssetsLazyList,
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: int,
        clear_dataset_cache: bool,
    ) -> ErrorRecap:
        pass
