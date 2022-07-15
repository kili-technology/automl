"""
This files contains types for annotations, but those types should be used only for label error.
"""
from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel
from shapely.geometry import Polygon
from typing_extensions import Literal

from kiliautoml.utils.type import CategoryIdT


class PositionT(BaseModel):
    ...


class PointT(BaseModel):
    x: float
    y: float


class SemanticPositionT(PositionT, BaseModel):
    points: List[PointT]


class BBoxPositionT(SemanticPositionT, BaseModel):
    ...


class NERPositionT(PositionT, BaseModel):
    first_token: int
    last_token: int
    page: int


class AnnotationStandardizedT(BaseModel, ABC):
    confidence: float
    category_id: CategoryIdT
    position: PositionT

    @abstractmethod
    def iou(self, position) -> float:
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


class AnnotationStandardizedBboxT(AnnotationStandardizedT, BaseModel):
    position: BBoxPositionT

    def iou(self, position: BBoxPositionT):
        return iou_polygons(self.position.points, position.points)


class AnnotationStandardizedNERT(AnnotationStandardizedT, BaseModel):
    position: NERPositionT

    def iou(self, position: BBoxPositionT) -> float:
        ...


class AssetStandardizedAnnotationsT(BaseModel):
    annotations: List[AnnotationStandardizedT]
    externalId: str


ErrorT = Literal["omission", "hallucination", "misclassification", "imprecise"]


def find_label_errors(
    predicted_annotations: AssetStandardizedAnnotationsT,
    manual_annotations: AssetStandardizedAnnotationsT,
):
    """We compare the prediction of the model and the manual annotations."""
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
            else:
                print("position not matching")

        if len(corresponding_annotation) == 0:
            print(
                "Either the model has forgotten something or the labeler has annotated something"
                " which should not be annotated."
            )
        elif len(corresponding_annotation) > 1:
            print("The model has found multiple objects.")
        else:
            print("ras")
