import pathlib
from dataclasses import dataclass
from typing import List, Optional, TypeVar

from typing_extensions import TypedDict

from kiliautoml.utils.helper_label_error import ErrorRecap
from kiliautoml.utils.helpers import set_default
from kiliautoml.utils.path import Path
from kiliautoml.utils.type import (
    AdditionalTrainingArgsT,
    AssetsLazyList,
    ContentInputT,
    EvalResultsT,
    InputTypeT,
    JobNameT,
    JobPredictions,
    JobT,
    MLBackendT,
    MLTaskT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
    ToolT,
)


class BaseInitArgs(TypedDict):
    """Common to all modalities"""

    job: JobT
    job_name: JobNameT
    model_name: Optional[ModelNameT]
    project_id: ProjectIdT
    ml_backend: MLBackendT
    api_key: str
    api_endpoint: Optional[str]
    title: str


class BaseTrainArgs(TypedDict):
    """Common to all modalities"""

    assets: AssetsLazyList
    local_dataset_dir: Optional[pathlib.Path]
    epochs: int
    batch_size: int
    clear_dataset_cache: bool
    disable_wandb: bool


class ModelTrainArgs(TypedDict):
    """Used only for some modalities"""

    additional_train_args_hg: AdditionalTrainingArgsT
    additional_train_args_yolo: AdditionalTrainingArgsT


class BaseEvaluateArgs(TypedDict):
    """Common to all modalities"""

    assets: AssetsLazyList
    batch_size: int
    clear_dataset_cache: bool
    model_path: Optional[str]
    from_project: Optional[ProjectIdT]


class ModelEvaluateArgs(TypedDict):
    """Used only for some modalities"""

    additional_train_args_hg: AdditionalTrainingArgsT


class BasePredictArgs(TypedDict):
    """Common to all modalities"""

    assets: AssetsLazyList
    model_path: Optional[str]
    from_project: Optional[ProjectIdT]
    batch_size: int
    clear_dataset_cache: bool


class BaseLabelErrorsArgs(TypedDict):
    """Common to all modalities"""

    assets: AssetsLazyList
    cv_n_folds: int
    epochs: int
    batch_size: int
    clear_dataset_cache: bool


T = TypeVar("T")  # Declare type variable


@dataclass
class ModelConditionsRequested:
    input_type: InputTypeT
    ml_task: MLTaskT
    content_input: ContentInputT
    tools: List[ToolT]
    ml_backend: Optional[MLBackendT]
    model_name: Optional[ModelNameT]
    model_repository: Optional[ModelRepositoryT]


@dataclass
class ModelConditions:
    input_type: InputTypeT
    ml_task: MLTaskT
    content_input: ContentInputT
    possible_ml_backend: List[MLBackendT]
    advised_model_names: List[ModelNameT]
    model_repository: ModelRepositoryT
    tools: Optional[List[ToolT]]

    @staticmethod
    def _check_compatible(request: Optional[T], list_ok: List[T], param_name: str) -> None:
        if request and request not in list_ok:
            raise ValueError(
                f"You requested {param_name} {request} but only {list_ok} is available."
            )

    def _check_compatible_model(self, model_name: Optional[ModelNameT]) -> None:
        self._check_compatible(model_name, self.advised_model_names, "model_name")

    def is_compatible(self, cdt_requested: ModelConditionsRequested) -> bool:
        strict_conditions = (
            self.input_type == cdt_requested.input_type
            and self.ml_task == cdt_requested.ml_task
            and self.content_input == cdt_requested.content_input
        )
        tools_ok = (
            self.tools is None
            or cdt_requested.tools is None
            or set(cdt_requested.tools).issubset(set(self.tools))
        )
        if strict_conditions and tools_ok:
            # We then check the loose conditions
            self._check_compatible(cdt_requested.ml_backend, self.possible_ml_backend, "ml_backend")
            self._check_compatible_model(cdt_requested.model_name)
            self._check_compatible(
                cdt_requested.model_repository, [self.model_repository], "model_repository"
            )
            return True
        else:
            return False


class KiliBaseModel:
    model_conditions: ModelConditions

    def __init__(self, base_init_args: BaseInitArgs) -> None:
        self.job = base_init_args["job"]
        self.job_name = base_init_args["job_name"]
        self.model_name: ModelNameT = (
            base_init_args["model_name"] or self.model_conditions.advised_model_names[0]
        )
        self.project_id = base_init_args["project_id"]

        self.model_repository_dir = Path.model_repository_dir(
            base_init_args["project_id"],
            base_init_args["job_name"],
            self.model_conditions.model_repository,
        )

        self.ml_backend: MLBackendT = set_default(
            base_init_args["ml_backend"],
            self.model_conditions.possible_ml_backend[0],
            "ml_backend",
            self.model_conditions.possible_ml_backend,
        )
        self.api_key = base_init_args["api_key"]
        self.api_endpoint = base_init_args["api_endpoint"]
        self.title = base_init_args["title"]

        self.fill_model_name(base_init_args=base_init_args)

    def fill_model_name(self, base_init_args: BaseInitArgs) -> None:
        self.model_name = set_default(
            base_init_args["model_name"],
            self.model_conditions.advised_model_names[0],
            "model_name",
            self.model_conditions.advised_model_names,
        )

    def train(
        self,
        *,
        assets: AssetsLazyList,
        local_dataset_dir: Optional[pathlib.Path],
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
        disable_wandb: bool,
        model_train_args: ModelTrainArgs,
    ) -> EvalResultsT:
        ...

    def eval(
        self,
        *,
        assets: AssetsLazyList,
        batch_size: int,
        clear_dataset_cache: bool,
        model_path: Optional[str],
        from_project: Optional[ProjectIdT],
    ) -> EvalResultsT:
        ...

    def predict(
        self,
        *,
        assets: AssetsLazyList,
        model_path: Optional[str],
        from_project: Optional[ProjectIdT],
        batch_size: int,
        clear_dataset_cache: bool,
    ) -> JobPredictions:
        ...

    def find_errors(
        self,
        *,
        assets: AssetsLazyList,
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
    ) -> ErrorRecap:
        ...
