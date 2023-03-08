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
    KiliBaseModel,
    ModelConditionsRequested,
)


def auto_get_model_class(
    condition_requested: ModelConditionsRequested,
) -> Type[KiliBaseModel]:
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


def auto_get_instantiated_model(
    condition_requested: ModelConditionsRequested, base_init_args: BaseInitArgs
) -> KiliBaseModel:
    return auto_get_model_class(condition_requested)(base_init_args=base_init_args)
