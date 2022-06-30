from typing import Any, Dict, List

from typing_extensions import Literal, TypedDict

AssetStatusT = Literal["TODO", "ONGOING", "LABELED", "TO_REVIEW", "REVIEWED"]
LabelTypeT = Literal["PREDICTION", "DEFAULT", "AUTOSAVE", "REVIEW", "INFERENCE"]
CommandT = Literal["train", "predict", "label_errors", "prioritize"]
LabelMergeStrategyT = Literal["last", "first"]


AnnotationsT = Any


class CategoryT(TypedDict):
    name: str
    confidence: int  # between 0 and 100


CategoriesT = List[CategoryT]


class JsonResponseT(TypedDict):
    annotations: AnnotationsT
    categories: CategoriesT


class LabelT(TypedDict):
    jsonResponse: JsonResponseT
    createdAt: str
    labelType: LabelTypeT


class AssetT(TypedDict):
    labels: List[LabelT]
    id: str
    externalId: str
    content: Any
    status: AssetStatusT


JobT = Dict[str, Any]
JobsT = Dict[str, JobT]
AdditionalTrainingArgsT = Dict[str, Any]
DictTrainingInfosT = Dict[str, Any]
