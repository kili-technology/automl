from typing import Any, Dict, List

from traitlets import Bool
from typing_extensions import Literal, TypedDict

from kiliautoml.utils.constants import MLTaskT

AssetStatusT = Literal["TODO", "ONGOING", "LABELED", "TO_REVIEW", "REVIEWED"]
LabelTypeT = Literal["PREDICTION", "DEFAULT", "AUTOSAVE", "REVIEW", "INFERENCE"]
CommandT = Literal["train", "predict", "label_errors", "prioritize"]
LabelMergeStrategyT = Literal["last", "first"]


AnnotationsT = Any


CategoryNameT = str
CategoryIdT = str  # camelCase with first letter in minuscule


class CategoryT(TypedDict):
    name: CategoryIdT
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


class OntologyCategory(TypedDict):
    children: Any
    name: CategoryNameT
    person: str
    color: str
    id: int


OntologyCategories = Dict[CategoryIdT, OntologyCategory]


class JobT(TypedDict):
    content: Dict[Literal["categories"], OntologyCategories]
    instruction: str
    isChild: Bool
    tools: Any  # example: ["semantic"],
    mlTask: MLTaskT
    models: Any  # example: {"interactive-segmentation": {"job": "SEMANTIC_JOB_MARKER"}},
    isVisible: Bool
    required: int
    isNew: Bool


JobsT = Dict[str, JobT]
AdditionalTrainingArgsT = Dict[str, Any]
