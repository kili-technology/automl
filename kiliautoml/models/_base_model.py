from abc import abstractmethod, ABCMeta
from typing import List, Dict, Optional

from kiliautoml.utils.helpers import JobPredictions
from kiliautoml.utils.constants import ModelFramework, ModelFrameworkT, ModelNameT


class BaseModel(metaclass=ABCMeta):
    def __init__(self) -> None:
        # internal state attributes
        self.model_framework: ModelFrameworkT = ModelFramework.PyTorch  # type: ignore

    @abstractmethod
    def train(
        self,
        assets: List[Dict],
        job: Dict,
        job_name: str,
        model_name: Optional[ModelNameT],
        clear_dataset_cache: bool = False,
    ):
        pass

    @abstractmethod
    def predict(
        self,
        assets: List[Dict],
        model_path: Optional[str],
        job_name: str,
        verbose: int = 0,
    ) -> JobPredictions:
        pass
