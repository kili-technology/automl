"""
This files contains types for annotations, but those types should be used only for label error.
"""
from dataclasses import dataclass
from typing import List

from shapely.geometry import Polygon
from typing_extensions import Literal, TypedDict

from kiliautoml.utils.type import CategoryIdT


class PositionT(TypedDict):
    ...


class PointT(TypedDict):
    x: float
    y: float


class SemanticPositionT(PositionT):
    points: List[PointT]


class BBoxPositionT(SemanticPositionT):
    ...


class NERPositionT(PositionT):
    first_token: int
    last_token: int
    page: int


class AnnotationStandardizedT:
    def __init__(
        self,
        confidence: float,
        category_id: CategoryIdT,
        position: PositionT,
    ):
        self.confidence = confidence
        self.category_id = category_id
        self.position = position


class AnnotationStandardizedSemanticT(AnnotationStandardizedT):
    def __init__(
        self,
        confidence: float,
        category_id: CategoryIdT,
        position: SemanticPositionT,
    ):
        super().__init__(confidence, category_id, position)
        self.position = position


class AnnotationStandardizedBboxT(AnnotationStandardizedSemanticT):
    def __init__(self, confidence: float, category_id: CategoryIdT, position: BBoxPositionT):
        self.confidence = confidence
        self.category_id = category_id
        self.position = position


class AnnotationStandardizedNERT(AnnotationStandardizedT):
    def __init__(self, confidence: float, category_id: CategoryIdT, position: NERPositionT):
        self.confidence = confidence
        self.category_id = category_id
        self.position = position


@dataclass
class AssetStandardizedAnnotationsT:
    annotations: List[AnnotationStandardizedT]
    externalId: str


def _iou_ner(
    annotation_ner_1: AnnotationStandardizedNERT, annotation_ner_2: AnnotationStandardizedNERT
) -> float:
    ...


def _iou_semantic(
    annotation_semantic_1: AnnotationStandardizedSemanticT,
    annotation_semantic_2: AnnotationStandardizedSemanticT,
) -> float:
    polygon1 = Polygon(annotation_semantic_1.position["points"])  # type:ignore
    polygon2 = Polygon(annotation_semantic_2.position["points"])  # type:ignore
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union
    return iou


def _iou_bbox(
    annotation_bbox_1: AnnotationStandardizedBboxT, annotation_bbox_2: AnnotationStandardizedBboxT
) -> float:
    return _iou_semantic(annotation_bbox_1, annotation_bbox_2)


def compute_iou(
    annotation_1: AnnotationStandardizedT, annotation_2: AnnotationStandardizedT
) -> float:
    # TypedDict
    if isinstance(annotation_1, AnnotationStandardizedNERT) and isinstance(
        annotation_2, AnnotationStandardizedNERT
    ):
        return _iou_ner(annotation_1, annotation_2)
    elif isinstance(annotation_1, AnnotationStandardizedBboxT) and isinstance(
        annotation_2, AnnotationStandardizedBboxT
    ):
        return _iou_bbox(annotation_1, annotation_2)
    elif isinstance(annotation_1, AnnotationStandardizedSemanticT) and isinstance(
        annotation_2, AnnotationStandardizedSemanticT
    ):
        return _iou_semantic(annotation_1, annotation_2)
    else:
        raise TypeError()


ErrorT = Literal["omission", "hallucination", "misclassification", "imprecise"]


def find_label_errors(
    predicted_annotations: AssetStandardizedAnnotationsT,
    manual_annotations: AssetStandardizedAnnotationsT,
):
    """We compare the prediction of the model and the manual annotations."""
    for manual_annotation in manual_annotations.annotations:

        corresponding_annotation = []
        for predicted_annotation in predicted_annotations.annotations:
            iou = compute_iou(manual_annotation, predicted_annotation)
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
