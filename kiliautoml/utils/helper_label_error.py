"""
- Rajouter des types
- Créer des unnitests
- ecrire les fonctions iou
- Tester sur les projets.
"""
from abc import ABC
from typing import List, Tuple

from typing_extensions import Literal, TypedDict

from kiliautoml.utils.type import CategoryIdT


class PositionT(TypedDict):
    ...


class PointT(TypedDict):
    x: float
    y: float


class SemanticPositionT(PositionT):
    points: List[PointT]


class BBoxPositionT(PositionT):
    points: Tuple[PointT, PointT]


class NERPositionT(PositionT):
    first_token: int
    last_token: int
    page: int


class AnnotationT(ABC):
    def __init__(self, confidence: float, category_id: CategoryIdT, position: PositionT):
        self.confidence = confidence
        self.category_id = category_id
        self.position = position


class AnnotationSemanticT(AnnotationT):
    def __init__(self, confidence: float, category_id: CategoryIdT, position: SemanticPositionT):
        self.confidence = confidence
        self.category_id = category_id
        self.position = position


class AnnotationBboxT(AnnotationT):
    def __init__(self, confidence: float, category_id: CategoryIdT, position: BBoxPositionT):
        self.confidence = confidence
        self.category_id = category_id
        self.position = position


class AnnotationNERT(AnnotationT):
    def __init__(self, confidence: float, category_id: CategoryIdT, position: NERPositionT):
        self.confidence = confidence
        self.category_id = category_id
        self.position = position


AssetAnnotationsT = List[AnnotationT]


def _iou_ner(annotation_ner_1: AnnotationNERT, annotation_ner_2: AnnotationNERT) -> float:
    ...


def _iou_bbox(annotation_bbox_1: AnnotationBboxT, annotation_bbox_2: AnnotationBboxT) -> float:
    ...


def _iou_semantic(
    annotation_semantic_1: AnnotationSemanticT, annotation_semantic_2: AnnotationSemanticT
) -> float:
    ...


def compute_iou(annotation_1: AnnotationT, annotation_2: AnnotationT) -> float:
    if type(annotation_1) != type(annotation_2):
        raise TypeError()

    if isinstance(annotation_1, AnnotationNERT):
        return _iou_ner(annotation_1, annotation_2)  # type:ignore
    elif isinstance(annotation_1, AnnotationBboxT):
        return _iou_bbox(annotation_1, annotation_2)  # type:ignore

    elif isinstance(annotation_1, AnnotationSemanticT):
        return _iou_semantic(annotation_1, annotation_2)  # type:ignore
    else:
        raise TypeError()


ErrorT = Literal["omission", "hallucination", "misclassification", "imprecise"]


def find_label_errors(
    predicted_annotations: AssetAnnotationsT, manual_annotations: AssetAnnotationsT
):
    """We compare the prediction of the model and the manual annotations."""
    for manual_annotation in manual_annotations:

        corresponding_annotation = []
        for predicted_annotation in predicted_annotations:
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
