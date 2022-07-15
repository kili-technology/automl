from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from typing_extensions import Literal, TypedDict

from kiliautoml.utils.helpers import kili_print

AssetStatusT = Literal["TODO", "ONGOING", "LABELED", "TO_REVIEW", "REVIEWED"]
LabelTypeT = Literal["PREDICTION", "DEFAULT", "AUTOSAVE", "REVIEW", "INFERENCE"]
CommandT = Literal["train", "predict", "label_errors", "prioritize"]
LabelMergeStrategyT = Literal["last", "first"]
ContentInputT = Literal["checkbox", "radio"]
InputTypeT = Literal["IMAGE", "TEXT"]
ModelFrameworkT = Literal["pytorch", "tensorflow"]
ModelRepositoryT = Literal["huggingface", "ultralytics", "torchvision", "detectron2"]
MLTaskT = Literal["CLASSIFICATION", "NAMED_ENTITIES_RECOGNITION", "OBJECT_DETECTION"]
ToolT = Literal["rectangle", "semantic", "polygon"]
ModelNameT = Literal[
    "bert-base-multilingual-cased",
    "distilbert-base-uncased",
    "distilbert-base-cased",
    "efficientnet_b0",
    "resnet50",
    "ultralytics/yolov5",
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
]


CategoryNameT = str
CategoryIdT = str  # camelCase with first letter in minuscule
JobNameT = str
ProjectIdT = str


class CategoryT(TypedDict):
    name: CategoryIdT
    confidence: int  # between 0 and 100


CategoriesT = List[CategoryT]

# ### ANNNOTATIONS #####################################

# KILI Polygon Semantic Format


class NormalizedVertice(TypedDict):
    x: float
    y: float


class NormalizedVertices(TypedDict):
    normalizedVertices: List[NormalizedVertice]


class KiliSemantic(TypedDict):
    boundingPoly: List[NormalizedVertices]  # len(self.boundingPoly) == 1
    mid: str
    type: Literal["semantic"]
    categories: List[CategoryT]


# BBOX


class PointT(TypedDict):
    x: float
    y: float


class BoundingPolyT(TypedDict):
    normalizedVertices: List[PointT]


class KiliBBox(TypedDict):
    boundingPoly: List[BoundingPolyT]
    type: str
    categories: List[CategoryT]


# KILI NER Format


class KiliNer(TypedDict):
    beginOffset: int
    content: str
    endOffset: int
    categories: CategoriesT


# KILI Text and Image Classification Format


class JsonResponseBaseT(TypedDict):
    """Compatible with JsonResponseSemanticT, ..."""

    ...


class JsonResponseSemanticT(JsonResponseBaseT, TypedDict):
    annotations: List[KiliSemantic]


class JsonResponseBboxT(JsonResponseBaseT, TypedDict):
    annotations: List[KiliBBox]


class JsonResponseNERT(JsonResponseBaseT, TypedDict):
    annotations: List[KiliNer]


class JsonResponseClassification(JsonResponseBaseT, TypedDict):
    categories: CategoriesT


# ### KILI TYPING


JsonResponseT = Dict[JobNameT, JsonResponseBaseT]


class LabelT(TypedDict):
    jsonResponse: JsonResponseT
    createdAt: str
    labelType: LabelTypeT


class AssetT(BaseModel):
    labels: List[LabelT]
    id: str
    externalId: str
    content: Any
    status: AssetStatusT

    def _get_annotations(self, job_name: JobNameT) -> JsonResponseBaseT:
        return self.labels[0]["jsonResponse"][job_name]

    def get_annotations_ner(self, job_name: JobNameT) -> JsonResponseNERT:
        return self._get_annotations(job_name)  # type:ignore

    def get_annotations_bbox(self, job_name: JobNameT) -> JsonResponseBboxT:
        return self._get_annotations(job_name)  # type:ignore

    def get_annotations_semantic(self, job_name: JobNameT) -> JsonResponseSemanticT:
        return self._get_annotations(job_name)  # type:ignore

    def get_annotations_classification(self, job_name: JobNameT) -> JsonResponseClassification:
        return self._get_annotations(job_name)  # type:ignore


class OntologyCategoryT(TypedDict):
    children: Any
    name: CategoryNameT
    person: str
    color: str
    id: int


OntologyCategoriesT = Dict[CategoryIdT, OntologyCategoryT]


class JobT(TypedDict):
    content: Dict[Literal["categories"], OntologyCategoriesT]  # Is this general?
    instruction: str
    isChild: bool
    tools: Any  # example: ["semantic"],
    mlTask: MLTaskT
    models: Any  # example: {"interactive-segmentation": {"job": "SEMANTIC_JOB_MARKER"}},
    isVisible: bool
    required: int
    isNew: bool


JobsT = Dict[JobNameT, JobT]

# AUTOML IDIOSYNCRATIC SPECIFIC TYPING


class JobPredictions:
    def __init__(
        self,
        job_name: JobNameT,
        external_id_array: List[str],
        json_response_array: List[Dict[JobNameT, JsonResponseBaseT]],
        model_name_array: List[str],
        predictions_probability: List[float],
    ):
        self.job_name = job_name
        self.external_id_array = external_id_array
        self.json_response_array = json_response_array
        self.model_name_array = model_name_array
        self.predictions_probability = predictions_probability

        n_assets = len(external_id_array)

        # assert all lists are compatible
        same_len = n_assets == len(json_response_array)
        assert same_len, "external_id_array and json_response_array must have the same length"

        same_len = n_assets == len(model_name_array)
        assert same_len, "external_id_array and model_name_array must have the same length"

        same_len = n_assets == len(predictions_probability)
        assert same_len, "external_id_array and predictions_probability must have the same length"

        # assert no duplicates
        assert (
            len(set(external_id_array)) == n_assets
        ), "external_id_array must not contain duplicates"

        kili_print(
            f"JobPredictions: {n_assets} predictions successfully created for job {job_name}."
        )

    def __repr__(self):
        return f"JobPredictions(job_name={self.job_name}, nb_assets={len(self.external_id_array)})"


AdditionalTrainingArgsT = Dict[str, Any]
DictTrainingInfosT = Dict[str, Any]


class ModelMetricT(TypedDict):
    overall: float
    by_category: Optional[List[float]]
