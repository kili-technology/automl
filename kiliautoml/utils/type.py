from typing import Any, Dict, Iterable, List, NewType, Optional

from kili.client import Kili
from more_itertools import chunked
from pydantic import BaseModel
from typing_extensions import Literal, TypedDict

AssetStatusT = Literal["TODO", "ONGOING", "LABELED", "TO_REVIEW", "REVIEWED"]
LabelTypeT = Literal["AUTOSAVE", "DEFAULT", "PREDICTION", "INFERENCE", "REVIEW"]
CommandT = Literal["advise", "train", "predict", "label_errors", "prioritize"]
LabelMergeStrategyT = Literal["last", "first"]
ContentInputT = Literal["checkbox", "radio"]
InputTypeT = Literal["IMAGE", "TEXT"]
MLBackendT = Literal["pytorch", "tensorflow"]
ParityFilterT = Literal["none", "keep-even", "keep-uneven"]
ModelRepositoryT = Literal["huggingface", "ultralytics", "torchvision", "detectron2"]
MLTaskT = Literal["CLASSIFICATION", "NAMED_ENTITIES_RECOGNITION", "OBJECT_DETECTION"]
ToolT = Literal["rectangle", "semantic", "polygon"]
VerboseLevelT = Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]


CategoryNameT = NewType("CategoryNameT", str)
CategoryIdT = NewType("CategoryIdT", str)  # camelCase with first letter in minuscule
JobNameT = NewType("JobNameT", str)
ProjectIdT = NewType("ProjectIdT", str)
AssetExternalIdT = NewType("AssetExternalIdT", str)
AssetIdT = NewType("AssetIdT", str)
ModelNameT = NewType("ModelNameT", str)


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


class KiliSemanticAnnotation(TypedDict):
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


class KiliBBoxAnnotation(TypedDict):
    boundingPoly: List[BoundingPolyT]
    type: str
    categories: List[CategoryT]


# KILI NER Format


class KiliNerAnnotation(TypedDict):
    beginOffset: int
    content: str
    endOffset: int
    categories: CategoriesT


# KILI Text and Image Classification Format


class JsonResponseBaseT(TypedDict):
    """Compatible with JsonResponseSemanticT, ..."""

    ...


class JsonResponseSemanticT(JsonResponseBaseT, TypedDict):
    annotations: List[KiliSemanticAnnotation]


class JsonResponseBboxT(JsonResponseBaseT, TypedDict):
    annotations: List[KiliBBoxAnnotation]


class JsonResponseNERT(JsonResponseBaseT, TypedDict):
    annotations: List[KiliNerAnnotation]


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
    id: AssetIdT
    externalId: AssetExternalIdT
    content: Any  # type:ignore

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

    def has_asset_for(self, job_name: JobNameT):
        return job_name in self.labels[0]["jsonResponse"] and self._get_annotations(job_name)


class OntologyCategoryT(TypedDict):
    children: Any
    name: CategoryNameT
    person: str
    color: str
    id: int


OntologyCategoriesT = Dict[CategoryIdT, OntologyCategoryT]


class ContentT(TypedDict):
    categories: OntologyCategoriesT
    input: ContentInputT


class JobT(TypedDict):
    content: ContentT
    instruction: str
    isChild: bool
    tools: List[ToolT]
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
        external_id_array: List[AssetExternalIdT],
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

    def __repr__(self):
        return f"JobPredictions(job_name={self.job_name}, nb_assets={len(self.external_id_array)})"


AdditionalTrainingArgsT = Dict[str, Any]
DictTrainingInfosT = Dict[str, Any]


class ModelMetricT(TypedDict):
    overall: float
    by_category: Optional[List[float]]


class PartialAsset(TypedDict):
    content: str
    id: str


class AssetsLazyList:
    """This class enables to iterate on the assets with and without Lazy refreshing.

    Lazy refreshing is used for accessing the asset_content which contains a signed url with
    expiration date of 5 minutes.
    """

    def __init__(self, assets: List[AssetT]):
        self.assets = assets
        self.counter = -1

    def iter_refreshed_asset(self, kili: Kili, project_id: ProjectIdT) -> Iterable[AssetT]:
        """Use this iterator if you need to access the assets 'content'"""
        batch = 20
        for assets in chunked(self.assets, batch):
            _ = kili.assets(
                project_id=project_id,
                asset_id_in=[asset.id for asset in assets],
                fields=["content", "id"],
                as_generator=False,
            )
            partial_assets: List[PartialAsset] = _  # type: ignore
            partial_assets = sorted(partial_assets, key=lambda d: d["id"])
            assets = sorted(assets, key=lambda d: d.id)
            for asset, partial_asset in zip(assets, partial_assets):
                asset.content = partial_asset["content"]
                yield asset

    def __len__(self):
        return len(self.assets)

    def __iter__(self):
        self.counter = -1
        return self

    def __next__(self):
        self.counter += 1
        if self.counter > len(self.assets) - 1:
            self.counter = 0
            raise StopIteration
        else:
            return self.assets[self.counter]

    def __getitem__(self, i):
        return self.assets.__getitem__(i)
