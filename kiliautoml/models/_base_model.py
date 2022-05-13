from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional

from kiliautoml.utils.constants import ModelFrameworkT, ModelNameT
from kiliautoml.utils.helpers import JobPredictions


class BaseModel(metaclass=ABCMeta):
    def __init__(self) -> None:
        # internal state attributes
        self.model_framework: ModelFrameworkT = "pytorch"

    @abstractmethod
    def train(
        self,
        assets: List[Dict],
        job: Dict,
        job_name: str,
        model_name: Optional[ModelNameT],
        clear_dataset_cache: bool = False,
        disable_wandb: bool = False,
    ):
        pass

    @abstractmethod
    def predict(
        self,
        assets: List[Dict],
        model_path: Optional[str],
        from_project: Optional[str],
        job_name: str,
        verbose: int = 0,
    ) -> JobPredictions:
        pass
