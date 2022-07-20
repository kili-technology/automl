"""
This files contains types for annotations, but those types should be used only for label error.
"""
from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, validator
from shapely.geometry import Polygon
from typing_extensions import Literal

from kiliautoml.utils.type import (
    AssetExternalIdT,
    AssetIdT,
    AssetT,
    CategoryIdT,
    JobNameT,
    JsonResponseBaseT,
    JsonResponseNERT,
    JsonResponseT,
    KiliBBoxAnnotation,
    KiliNerAnnotation,
    KiliSemanticAnnotation,
    MLTaskT,
    NormalizedVertice,
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


def iou_polygons(points1, points2):
    polygon1 = Polygon(points1)
    polygon2 = Polygon(points2)
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union
    return iou


class AnnotationStandardizedSemanticT(AnnotationStandardizedT, BaseModel):
    position: SemanticPositionT

    def iou(self, position: SemanticPositionT):
        return iou_polygons(self.position.points, position.points)

    @classmethod
    def from_annotation(
        cls, annotation: KiliSemanticAnnotation
    ) -> "AnnotationStandardizedSemanticT":
        position = SemanticPositionT(points=annotation["boundingPoly"][0]["normalizedVertices"])
        return cls(
            confidence=annotation["categories"][0]["confidence"],
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
            confidence=annotation["categories"][0]["confidence"],
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
            confidence=annotation["categories"][0]["confidence"],
            category_id=annotation["categories"][0]["name"],
            position=position,
            ml_task="NAMED_ENTITIES_RECOGNITION",
        )


class AssetStandardizedAnnotationsT(BaseModel):
    annotations: List[AnnotationStandardizedT]
    externalId: str


LabelingErrorTypeT = Literal["omission", "hallucination", "misclassification", "imprecise"]


class LabelingError(BaseModel):
    error_type: LabelingErrorTypeT
    error_probability: float


def find_label_errors_for_one_asset(
    predicted_annotations: AssetStandardizedAnnotationsT,
    manual_annotations: AssetStandardizedAnnotationsT,
) -> List[LabelingError]:
    """We compare the prediction of the model and the manual annotations."""

    errors: List[LabelingError] = []
    for manual_annotation in manual_annotations.annotations:

        corresponding_annotation = []
        for predicted_annotation in predicted_annotations.annotations:
            iou = manual_annotation.iou(predicted_annotation)
            same_category = manual_annotation.category_id == predicted_annotation.category_id
            good_iou = iou > 0.9

            if good_iou and same_category:
                corresponding_annotation.append(predicted_annotation)
            elif good_iou and not same_category:
                print("Probably wrong category")
                errors.append(LabelingError(error_type="misclassification", error_probability=iou))
            else:
                print("position not matching")
                errors.append(LabelingError(error_type="imprecise", error_probability=1 - iou))

        if len(corresponding_annotation) == 0:
            print(
                "Either the model has forgotten something or the labeler has annotated something"
                " which should not be annotated."
            )
            errors.append(LabelingError(error_type="hallucination", error_probability=0.1))
        elif len(corresponding_annotation) > 1:
            print("The model has found multiple objects.")
        else:
            print("ras")

    return errors


def create_normalized_annotation(
    json_response: JsonResponseBaseT, ml_task: MLTaskT
) -> List[AnnotationStandardizedT]:
    if ml_task == "NAMED_ENTITIES_RECOGNITION":
        json_response_ner: JsonResponseNERT = json_response  # type:ignore
        return [
            AnnotationStandardizedNERT.from_annotation(kili_ner)
            for kili_ner in json_response_ner["annotations"]
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
) -> ErrorRecap:
    assert len(assets) == len(json_response_array)

    errors_by_asset: List[List[LabelingError]] = []

    for json_response, asset in zip(json_response_array, assets):

        json_response_base = asset._get_annotations(job_name)

        predicted_annotations = create_normalized_annotation(json_response[job_name], ml_task)
        manual_annotations = create_normalized_annotation(json_response_base, ml_task)

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
