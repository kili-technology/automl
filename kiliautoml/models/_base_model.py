from abc import ABCMeta, abstractmethod
from typing import List, Optional

from kiliautoml.utils.constants import (
    MLTaskT,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
)
from kiliautoml.utils.helpers import JobPredictions
from kiliautoml.utils.type import AssetT, JobT


class BaseModel(metaclass=ABCMeta):
    ml_task: MLTaskT  # type: ignore
    model_repository: ModelRepositoryT  # type: ignore

    def __init__(
        self,
        job: JobT,
        job_name: str,
        model_name: ModelNameT,
        model_framework: ModelFrameworkT,
    ) -> None:
        self.job = job
        self.job_name = job_name
        self.model_name = model_name
        self.model_framework: ModelFrameworkT = model_framework

    @abstractmethod
    def train(
        self,
        assets: List[AssetT],
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool = False,
        disable_wandb: bool = False,
        # verbose: int = 0,
    ) -> float:
        pass

    @abstractmethod
    def predict(
        self,
        assets: List[AssetT],
        model_path: str,
        from_project: Optional[str],
        batch_size: int,
        verbose: int = 0,
        clear_dataset_cache: bool = False,
    ) -> JobPredictions:
        pass

    @abstractmethod
    def find_errors(
        self,
        assets: List[AssetT],
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: int = 0,
        clear_dataset_cache: bool = False,
    ):
        pass
