"""
This files contains types for annotations, but those types should be used only for label error.
"""
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Type, Union

from pydantic import BaseModel, validator
from shapely.errors import TopologicalError
from shapely.geometry import Point, Polygon
from typing_extensions import Literal

from kiliautoml.utils.logging import OneTimePrinter
from kiliautoml.utils.type import (
    AssetExternalIdT,
    AssetIdT,
    AssetsLazyList,
    CategoryIdT,
    JobNameT,
    JsonResponseBaseT,
    JsonResponseT,
    KiliBBoxAnnotation,
    KiliNerAnnotation,
    KiliSemanticAnnotation,
    MLTaskT,
    NormalizedVertice,
    ToolT,
)


class PositionT(BaseModel):
    ...


class SemanticPositionT(PositionT, BaseModel):
    points: List[NormalizedVertice]

    @validator("points")
    def number_of_points(cls, points: List[NormalizedVertice]):
        _ = cls

        points = [
            NormalizedVertice(x=round(point["x"], 4), y=round(point["y"], 4)) for point in points
        ]
        # Delete duplicates
        points = [dict(t) for t in {tuple(d.items()) for d in points}]  # type:ignore

        if len(points) < 3:
            raise ValueError("Semantic annotation should contain at least 3 points.")
        return points


class BBoxPositionT(SemanticPositionT, BaseModel):
    @validator("points")
    def number_of_points(cls, points):
        _ = cls
        if len(points) != 4:
            raise ValueError("Bbox annotation should contain 4 points.")
        return points


class NERPositionT(PositionT, BaseModel):
    beginOffset: int
    content: str
    endOffset: int


class AnnotationStandardizedT(BaseModel, ABC):
    confidence: float
    category_id: CategoryIdT
    position: PositionT
    ml_task: MLTaskT

    @abstractmethod
    def iou(self, position) -> float:
        ...

    @validator("confidence")
    def confidence_probability(cls, confidence):
        _ = cls
        if confidence > 1:
            raise ValueError("Normalized confidence should be a probability")
        return confidence

    @classmethod
    @abstractmethod
    def from_annotation(cls, annotation) -> "AnnotationStandardizedT":
        ...


# TODO: Create a real logging class
one_time_printer = OneTimePrinter()


def iou_polygons(points1: List[NormalizedVertice], points2):
    polygon1 = Polygon([Point(p["x"], p["y"]) for p in points1])
    polygon2 = Polygon([Point(p["x"], p["y"]) for p in points2])
    try:
        intersect = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        iou = intersect / union
    except TopologicalError:
        one_time_printer("TopologicalError: The model is probably not trained enough")
        iou = 0
    return iou


def get_confidence(
    annotation: Union[KiliSemanticAnnotation, KiliBBoxAnnotation, KiliNerAnnotation]
) -> float:
    if "confidence" in annotation["categories"][0]:
        return annotation["categories"][0]["confidence"] / 100
    else:
        return 1


class InvalidAnnotation(Exception):
    ...


class AnnotationStandardizedSemanticT(AnnotationStandardizedT, BaseModel):
    position: SemanticPositionT

    def iou(self, position: SemanticPositionT):
        a = self.position.points
        if len(a) < 3:
            raise InvalidAnnotation("Polygon must contain more than 2 points.")
        return iou_polygons(a, position.points)

    @classmethod
    def from_annotation(
        cls, annotation: KiliSemanticAnnotation
    ) -> "AnnotationStandardizedSemanticT":
        position = SemanticPositionT(points=annotation["boundingPoly"][0]["normalizedVertices"])
        return cls(
            confidence=get_confidence(annotation),
            category_id=annotation["categories"][0]["name"],
            position=position,
            ml_task="OBJECT_DETECTION",
        )


class AnnotationStandardizedBboxT(AnnotationStandardizedT, BaseModel):
    position: BBoxPositionT

    def iou(self, position: BBoxPositionT):
        return iou_polygons(self.position.points, position.points)

    @classmethod
    def from_annotation(cls, annotation: KiliBBoxAnnotation) -> "AnnotationStandardizedBboxT":
        position = BBoxPositionT(points=annotation["boundingPoly"][0]["normalizedVertices"])
        return cls(
            confidence=get_confidence(annotation),
            category_id=annotation["categories"][0]["name"],
            position=position,
            ml_task="OBJECT_DETECTION",
        )


class AnnotationStandardizedNERT(AnnotationStandardizedT, BaseModel):
    position: NERPositionT

    def iou(self, position: BBoxPositionT) -> float:
        ...

    @classmethod
    def from_annotation(cls, annotation: KiliNerAnnotation) -> "AnnotationStandardizedNERT":
        position = NERPositionT(
            beginOffset=annotation["beginOffset"],
            content=annotation["content"],
            endOffset=annotation["endOffset"],
        )
        return cls(
            confidence=get_confidence(annotation),
            category_id=annotation["categories"][0]["name"],
            position=position,
            ml_task="NAMED_ENTITIES_RECOGNITION",
        )


class AssetStandardizedAnnotationsT(BaseModel):
    annotations: List[AnnotationStandardizedT]
    externalId: str

    def category_count(self):
        categories_count = {}

        for annotation in self.annotations:
            categories_count[annotation.category_id] = 0

        for annotation in self.annotations:
            categories_count[annotation.category_id] += 1

        return categories_count


def get_asset_standardized_class(ml_task: MLTaskT, tool: ToolT) -> Type[AnnotationStandardizedT]:
    if ml_task == "NAMED_ENTITIES_RECOGNITION":
        return AnnotationStandardizedNERT
    elif ml_task == "OBJECT_DETECTION" and tool == "rectangle":
        return AnnotationStandardizedBboxT
    elif ml_task == "OBJECT_DETECTION" and tool in ["polygon", "semantic"]:
        return AnnotationStandardizedSemanticT
    else:
        raise NotImplementedError


LabelingErrorTypeT = Literal["misclassification", "omission", "hallucination", "imprecise", "other"]


class LabelingError(BaseModel):
    error_type: LabelingErrorTypeT
    error_probability: float

    def __gt__(self, other):
        return self.error_probability > other.error_probability


def find_label_errors_for_one_asset(
    predicted_annotations: AssetStandardizedAnnotationsT,
    manual_annotations: AssetStandardizedAnnotationsT,
) -> List[LabelingError]:
    """We compare the prediction of the model and the manual annotations."""

    print("predicted annotations:", predicted_annotations.category_count())
    print("manual annotations:", manual_annotations.category_count())

    errors: List[LabelingError] = []
    prediction_to_manual: Dict[int, List[int]] = {
        k: [] for k in range(len(predicted_annotations.annotations))
    }
    for man_i, manual_ann in enumerate(manual_annotations.annotations):
        msg = f"Analysing {manual_ann.category_id}..."
        corresponding_perfect_annotation = []
        for pred_i, predicted_ann in enumerate(predicted_annotations.annotations):
            iou = manual_ann.iou(predicted_ann.position)
            same_category = manual_ann.category_id == predicted_ann.category_id
            good_iou = iou > 0.8

            only_touching = iou < 0.2
            if only_touching:
                # print("Probably 2 different objects on the Asset.")
                continue

            # The assets overlap significantly
            prediction_to_manual[pred_i].append(man_i)
            if good_iou and same_category:
                corresponding_perfect_annotation.append(predicted_ann)
                print(msg, "perfect annotation!")
            elif good_iou and not same_category:
                # Weird
                add_error(msg, errors, manual_ann, predicted_ann, iou, "misclassification")
            elif same_category:
                add_error(msg, errors, manual_ann, predicted_ann, iou, "imprecise")
            else:
                add_error(msg, errors, manual_ann, predicted_ann, iou, "other")

        if len(corresponding_perfect_annotation) > 1:
            print(
                "The model has found multiple objects. "
                "Sometimes this happens because the model is not trained enouth."
            )

    # Checking for Omissions
    ommissions = []
    for prediction_i, manuals in prediction_to_manual.items():
        if len(manuals) == 0:
            ann = predicted_annotations.annotations[prediction_i]
            ommissions.append(ann.category_id)
            errors.append(LabelingError(error_type="omission", error_probability=ann.confidence))
    print(f"Found omission: {Counter(ommissions)}.")

    print(f"Asset conclusion: We found {len(errors)} errors.")

    return errors


def add_error(
    msg,
    errors: List[LabelingError],
    manual_ann: AnnotationStandardizedT,
    predicted_ann: AnnotationStandardizedT,
    iou: float,
    error_type: LabelingErrorTypeT,
):
    if error_type in ["misclassification", "other"]:
        err = (
            f"{error_type}: Annotation {manual_ann.category_id} should be"
            f" {predicted_ann.category_id}"
        )
        error_probability = predicted_ann.confidence
    elif error_type == "imprecise":
        err = f"imprecise: Annotation {manual_ann.category_id} is imprecise, iou {iou}"
        error_probability = predicted_ann.confidence * (1 - iou)  # TODO: better formula
    else:
        raise ValueError

    print(msg, err, f"error_proba: {int(error_probability*100)}%")
    errors.append(LabelingError(error_type=error_type, error_probability=error_probability))


def create_normalized_annotation(
    json_response: JsonResponseBaseT, ml_task: MLTaskT, tool: ToolT
) -> List[AnnotationStandardizedT]:

    res = []
    for annotation in json_response["annotations"]:  # type:ignore
        try:
            res.append(get_asset_standardized_class(ml_task, tool).from_annotation(annotation))
        except ValueError as e:
            print(e)
    return res


class ErrorRecap(BaseModel):
    external_id_array: List[AssetExternalIdT]
    id_array: List[AssetIdT]
    errors_by_asset: List[List[LabelingError]]


def find_all_label_errors(
    assets: AssetsLazyList,
    json_response_array: List[JsonResponseT],
    external_id_array: List[AssetExternalIdT],
    job_name: JobNameT,
    ml_task: MLTaskT,
    tool: ToolT,
) -> ErrorRecap:
    assert len(assets) == len(json_response_array)

    errors_by_asset: List[List[LabelingError]] = []

    for json_response, asset in zip(json_response_array, assets):

        json_response_base = asset._get_annotations(job_name)

        predicted_annotations = create_normalized_annotation(json_response[job_name], ml_task, tool)
        manual_annotations = create_normalized_annotation(json_response_base, ml_task, tool)

        print("\nAsset externalId", asset.externalId)
        labeling_errors = find_label_errors_for_one_asset(
            predicted_annotations=AssetStandardizedAnnotationsT(
                annotations=predicted_annotations,
                externalId=asset.externalId,
            ),
            manual_annotations=AssetStandardizedAnnotationsT(
                annotations=manual_annotations,
                externalId=asset.externalId,
            ),
        )
        errors_by_asset.append(labeling_errors)

    return ErrorRecap(
        id_array=[asset.id for asset in assets],
        external_id_array=external_id_array,
        errors_by_asset=errors_by_asset,
    )
