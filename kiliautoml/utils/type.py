from typing import Any, Dict, List, Optional

from typing_extensions import Literal, TypedDict

AssetStatusT = Literal["TODO", "ONGOING", "LABELED", "TO_REVIEW", "REVIEWED"]
LabelTypeT = Literal["PREDICTION", "DEFAULT", "AUTOSAVE", "REVIEW", "INFERENCE"]
CommandT = Literal["train", "predict", "label_errors", "prioritize"]
LabelMergeStrategyT = Literal["last", "first"]
ContentInputT = Literal["checkbox", "radio"]
InputTypeT = Literal["IMAGE", "TEXT"]
ModelFrameworkT = Literal["pytorch", "tensorflow"]
ModelRepositoryT = Literal["huggingface", "ultralytics", "torchvision", "detectron2"]
MLTaskT = Literal["CLASSIFICATION", "NAMED_ENTITIES_RECOGNITION", "OBJECT_DETECTION"]
ToolT = Literal["rectangle", "semantic", "polygone"]
ModelNameT = Literal[
    "bert-base-multilingual-cased",
    "distilbert-base-uncased",
    "distilbert-base-cased",
    "efficientnet_b0",
    "resnet50",
    "ultralytics/yolov5",
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
]


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


class OntologyCategoryT(TypedDict):
    children: Any
    name: CategoryNameT
    person: str
    color: str
    id: int


OntologyCategoriesT = Dict[CategoryIdT, OntologyCategoryT]


class JobT(TypedDict):
    content: Dict[Literal["categories"], OntologyCategoriesT]
    instruction: str
    isChild: bool
    tools: Any  # example: ["semantic"],
    mlTask: MLTaskT
    models: Any  # example: {"interactive-segmentation": {"job": "SEMANTIC_JOB_MARKER"}},
    isVisible: bool
    required: int
    isNew: bool


JobsT = Dict[str, JobT]
AdditionalTrainingArgsT = Dict[str, Any]
DictTrainingInfosT = Dict[str, Any]


class ModelMetricT(TypedDict):
    overall: float
    by_category: Optional[List[float]]
