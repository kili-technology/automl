"""
This files contains types for annotations, but those types should be used only for label error.
"""
from abc import ABC, abstractmethod
from typing import List, Union

from pydantic import BaseModel, validator
from shapely.geometry import Point, Polygon
from typing_extensions import Literal

from kiliautoml.utils.type import (
    AssetExternalIdT,
    AssetIdT,
    AssetT,
    CategoryIdT,
    JobNameT,
    JsonResponseBaseT,
    JsonResponseBboxT,
    JsonResponseNERT,
    JsonResponseSemanticT,
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


class BBoxPositionT(SemanticPositionT, BaseModel):
    @validator("points")
    def number_of_points(cls, points):
        _ = cls
        if len(points) != 4:
            raise ValueError("Bbox should contain 4 points.")
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


def iou_polygons(points1: List[NormalizedVertice], points2):
    polygon1 = Polygon([Point(p["x"], p["y"]) for p in points1])
    polygon2 = Polygon([Point(p["x"], p["y"]) for p in points2])
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union
    return iou


def get_confidence(
    annotation: Union[KiliSemanticAnnotation, KiliBBoxAnnotation, KiliNerAnnotation]
) -> float:
    if "confidence" in annotation["categories"][0]:
        return annotation["categories"][0]["confidence"] / 100
    else:
        return 1


class AnnotationStandardizedSemanticT(AnnotationStandardizedT, BaseModel):
    position: SemanticPositionT

    def iou(self, position: SemanticPositionT):
        a = self.position.points
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


LabelingErrorTypeT = Literal["omission", "hallucination", "misclassification", "imprecise"]


class LabelingError(BaseModel):
    error_type: LabelingErrorTypeT
    error_probability: float


def find_label_errors_for_one_asset(
    predicted_annotations: AssetStandardizedAnnotationsT,
    manual_annotations: AssetStandardizedAnnotationsT,
) -> List[LabelingError]:
    """We compare the prediction of the model and the manual annotations."""

    print("\npredicted annotations:", predicted_annotations.category_count())
    print("manual annotations:", manual_annotations.category_count())

    errors: List[LabelingError] = []
    for manual_ann in manual_annotations.annotations:
        msg = f"Analysing {manual_ann.category_id}..."
        corresponding_perfect_annotation = []
        for predicted_ann in predicted_annotations.annotations:
            iou = manual_ann.iou(predicted_ann.position)
            same_category = manual_ann.category_id == predicted_ann.category_id
            good_iou = iou > 0.8

            if good_iou and same_category:
                corresponding_perfect_annotation.append(predicted_ann)
                print(msg, "perfect annotation!")
            elif good_iou and not same_category:
                add_error(msg, errors, manual_ann, predicted_ann, iou, "misclassification")
            elif iou < 0.1:
                pass
                # print("Probably 2 different objects on the Asset.")
            elif same_category:
                add_error(msg, errors, manual_ann, predicted_ann, iou, "imprecise")
            else:
                # Bad category bad iou, weird, let's say it's a misclassification
                add_error(msg, errors, manual_ann, predicted_ann, iou, "misclassification")

        if len(corresponding_perfect_annotation) == 0:
            # Hallucination must be very rare. Don't consider it for the moment.
            print("The model is probably not trained enough")
        elif len(corresponding_perfect_annotation) > 1:
            print(
                "The model has found multiple objects. "
                "Sometimes this happens because the model is not trained enouth."
            )
        else:
            pass
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
    if error_type == "misclassification":
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

    print(msg, err)
    errors.append(LabelingError(error_type=error_type, error_probability=error_probability))


def create_normalized_annotation(
    json_response: JsonResponseBaseT, ml_task: MLTaskT, tool: ToolT
) -> List[AnnotationStandardizedT]:
    if ml_task == "NAMED_ENTITIES_RECOGNITION":
        json_response_ner: JsonResponseNERT = json_response  # type:ignore
        return [
            AnnotationStandardizedNERT.from_annotation(kili_ner)
            for kili_ner in json_response_ner["annotations"]
        ]
    elif ml_task == "OBJECT_DETECTION" and tool == "rectangle":
        json_response_bbox: JsonResponseBboxT = json_response  # type:ignore
        return [
            AnnotationStandardizedBboxT.from_annotation(kili_bbox)
            for kili_bbox in json_response_bbox["annotations"]
        ]
    elif ml_task == "OBJECT_DETECTION" and tool in ["polygon", "semantic"]:
        json_response_semantic: JsonResponseSemanticT = json_response  # type:ignore
        return [
            AnnotationStandardizedSemanticT.from_annotation(kili_sem)
            for kili_sem in json_response_semantic["annotations"]
        ]
    else:
        raise NotImplementedError


class ErrorRecap(BaseModel):
    external_id_array: List[AssetExternalIdT]
    id_array: List[AssetIdT]
    errors_by_asset: List[List[LabelingError]]


def find_all_label_errors(
    assets: List[AssetT],
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
