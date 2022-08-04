from typing import List, Type

from kiliautoml.models import (
    Detectron2SemanticSegmentationModel,
    HuggingFaceNamedEntityRecognitionModel,
    HuggingFaceTextClassificationModel,
    PyTorchVisionImageClassificationModel,
    UltralyticsObjectDetectionModel,
)
from kiliautoml.models._base_model import (
    BaseInitArgs,
    BaseLabelErrorsArgs,
    BasePredictArgs,
    BaseTrainArgs,
    KiliBaseModel,
    ModalTrainArgs,
    ModelConditionsRequested,
)
from kiliautoml.utils.helper_label_error import ErrorRecap
from kiliautoml.utils.type import DictTrainingInfosT, JobPredictions


def get_appropriate_model(condition_requested: ModelConditionsRequested) -> Type[KiliBaseModel]:

    models: List[Type[KiliBaseModel]] = [
        Detectron2SemanticSegmentationModel,
        HuggingFaceNamedEntityRecognitionModel,
        HuggingFaceTextClassificationModel,
        PyTorchVisionImageClassificationModel,
        UltralyticsObjectDetectionModel,
    ]
    for model in models:
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
        modal_train_args: ModalTrainArgs,
    ) -> DictTrainingInfosT:
        return self.model.train(**base_train_args, modal_train_args=modal_train_args)

    def predict(self, *, base_predict_args: BasePredictArgs) -> JobPredictions:
        return self.model.predict(**base_predict_args)

    def find_errors(self, *, base_label_errors_args: BaseLabelErrorsArgs) -> ErrorRecap:
        return self.model.find_errors(**base_label_errors_args)
