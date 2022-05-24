# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
import os
from typing import Any, Dict, List, Optional

from kiliautoml.models._base_model import BaseModel
from kiliautoml.utils.constants import (
    HOME,
    MLTaskT,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
)
from kiliautoml.utils.helpers import (
    JobPredictions,
    get_last_trained_model_path,
    kili_print,
)
from kiliautoml.utils.path import Path
from kiliautoml.utils.type import AssetT, JobT
from kiliautoml.utils.ultralytics.train import ultralytics_train_yolov5

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def predict_object_detection(
    *,
    api_key: str,
    assets: List[AssetT],
    job_name: str,
    project_id: str,
    model_path: str,
    verbose: int,
    prioritization: bool,
    batch_size: int,
) -> JobPredictions:
    from kiliautoml.utils.ultralytics.predict_ultralytics import (
        ultralytics_predict_object_detection,
    )

    split_path = os.path.normpath(model_path).split(os.path.sep)
    model_repository = split_path[-7]
    kili_print(f"Model base repository: {model_repository}")
    if model_repository not in ["ultralytics"]:
        raise ValueError(f"Unknown model base repository: {model_repository}")

    model_framework: ModelFrameworkT = split_path[-5]  # type: ignore
    kili_print(f"Model framework: {model_framework}")
    if model_framework not in ["pytorch", "tensorflow"]:
        raise ValueError(f"Unknown model framework: {model_framework}")

    if model_repository == "ultralytics":
        job_predictions = ultralytics_predict_object_detection(
            api_key,
            assets,
            project_id,
            model_framework,
            model_path,
            job_name,
            verbose,
            batch_size,
            prioritization,
        )
    else:
        raise NotImplementedError

    return job_predictions


class UltralyticsObjectDetectionModel(BaseModel):

    ml_task: MLTaskT = "OBJECT_DETECTION"
    model_repository: ModelRepositoryT = "ultralytics"

    def __init__(
        self,
        *,
        project_id: str,
        job: JobT,
        job_name: str,
        model_name: ModelNameT,
        model_framework: ModelFrameworkT,
    ):
        BaseModel.__init__(
            self,
            job=job,
            job_name=job_name,
            model_name=model_name,
            model_framework=model_framework,
        )
        self.project_id = project_id

    def train(
        self,
        *,
        assets: List[AssetT],
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
        disable_wandb: bool,
        verbose: int,
        title: str,
        args_dict: Dict,  # type: ignore
        api_key: str,
    ):
        _ = verbose

        model_repository_dir = Path.model_repository_dir(
            HOME, self.project_id, self.job_name, self.model_repository
        )

        return ultralytics_train_yolov5(
            api_key=api_key,
            model_repository_dir=model_repository_dir,
            job=self.job,
            assets=assets,
            json_args=args_dict,
            epochs=epochs,
            model_framework=self.model_framework,
            clear_dataset_cache=clear_dataset_cache,
            title=title,
            disable_wandb=disable_wandb,
            batch_size=batch_size,
        )

    def predict(  # type: ignore
        self,
        *,
        assets: List[AssetT],
        model_path: Optional[str],
        from_project: Optional[str],
        batch_size: int,
        verbose: int,
        clear_dataset_cache: bool,
        api_key: str = "",
    ):
        _ = clear_dataset_cache

        project_id = from_project if from_project else self.project_id

        model_path_res = get_last_trained_model_path(
            project_id=project_id,
            job_name=self.job_name,
            project_path_wildcard=[
                "*",  # ultralytics or huggingface
                "model",
                "*",  # pytorch or tensorflow
                "*",  # date and time
                "*",  # title of the project, but already specified by project_id
                "exp",
                "weights",
            ],
            weights_filename="best.pt",
            model_path=model_path,
        )

        job_predictions = predict_object_detection(
            api_key=api_key,
            assets=assets,
            project_id=self.project_id,
            model_path=model_path if model_path else model_path_res,
            job_name=self.job_name,
            verbose=verbose,
            prioritization=False,
            batch_size=batch_size,
        )
        return job_predictions

    def find_errors(
        self,
        *,
        assets: List[AssetT],
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: int = 0,
        clear_dataset_cache: bool = False,
        api_key: str = "",
    ) -> Any:
        pass
