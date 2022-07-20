from abc import ABCMeta, abstractmethod
from typing import List, Optional

from typing_extensions import TypedDict

from kiliautoml.utils.helper_label_error import ErrorRecap
from kiliautoml.utils.type import (
    AssetT,
    DictTrainingInfosT,
    JobNameT,
    JobPredictions,
    JobT,
    MLTaskT,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
)


class BaseInitArgs(TypedDict):
    job: JobT
    job_name: JobNameT
    model_name: ModelNameT
    model_framework: ModelFrameworkT
    project_id: ProjectIdT


class BaseTrainArgs(TypedDict):
    assets: List[AssetT]
    epochs: int
    batch_size: int
    clear_dataset_cache: bool
    disable_wandb: bool
    verbose: int


class BaseModel(metaclass=ABCMeta):
    ml_task: MLTaskT  # type: ignore
    model_repository: ModelRepositoryT  # type: ignore

    def __init__(
        self,
        *,
        job: JobT,
        job_name: JobNameT,
        model_name: ModelNameT,
        model_framework: ModelFrameworkT,
        project_id: ProjectIdT,
    ) -> None:
        self.job = job
        self.job_name = job_name
        self.model_name = model_name
        self.model_framework: ModelFrameworkT = model_framework
        self.project_id: ProjectIdT = project_id

    @abstractmethod
    def train(
        self,
        *,
        assets: List[AssetT],
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
        assets: List[AssetT],
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
        assets: List[AssetT],
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: int,
        clear_dataset_cache: bool,
    ) -> ErrorRecap:
        pass
