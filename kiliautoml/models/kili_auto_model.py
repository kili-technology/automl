from typing import List, Optional, Type

from kiliautoml.models import (
    Detectron2SemanticSegmentationModel,
    HuggingFaceNamedEntityRecognitionModel,
    HuggingFaceTextClassificationModel,
    PyTorchVisionImageClassificationModel,
    UltralyticsObjectDetectionModel,
)
from kiliautoml.models._base_model import (
    BaseInitArgs,
    BaseTrainArgs,
    KiliBaseModel,
    ModelConditionsRequested,
)
from kiliautoml.utils.helper_label_error import ErrorRecap
from kiliautoml.utils.type import (
    AdditionalTrainingArgsT,
    AssetsLazyList,
    DictTrainingInfosT,
    JobPredictions,
    ProjectIdT,
)


def get_appropriate_model(condition_requested: ModelConditionsRequested) -> Type[KiliBaseModel]:

    models: List[Type[KiliBaseModel]] = [
        Detectron2SemanticSegmentationModel,
        HuggingFaceNamedEntityRecognitionModel,
        HuggingFaceTextClassificationModel,
        PyTorchVisionImageClassificationModel,
        UltralyticsObjectDetectionModel,
    ]
    for model in models:
        if model == PyTorchVisionImageClassificationModel:
            print(condition_requested)
        if model.model_conditions.is_compatible(condition_requested):
            return model
    raise NotImplementedError


class KiliAutoModel:
    def __init__(
        self, *, condition_requested: ModelConditionsRequested, base_init_args: BaseInitArgs
    ) -> None:

        Model = get_appropriate_model(condition_requested)
        self.model = Model(base_init_args=base_init_args)

    def train(
        self,
        *,
        base_train_args: BaseTrainArgs,
        additional_train_args_yolo: AdditionalTrainingArgsT,
        additional_train_args_hg: AdditionalTrainingArgsT,
    ) -> DictTrainingInfosT:
        return self.model.train(
            **base_train_args,
            additional_train_args_yolo=additional_train_args_yolo,
            additional_train_args_hg=additional_train_args_hg,
        )

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
        return self.model.predict(
            assets=assets,
            model_path=model_path,
            batch_size=batch_size,
            clear_dataset_cache=clear_dataset_cache,
            verbose=verbose,
            from_project=from_project,
        )

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
        return self.model.find_errors(
            assets=assets,
            cv_n_folds=cv_n_folds,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            clear_dataset_cache=clear_dataset_cache,
        )
