import os
from pathlib import Path
from typing import Dict, List

from typing_extensions import Literal, TypedDict

HOME = os.path.join(os.getenv("HOME"), ".cache", "kili", "automl")  # type: ignore

AUTOML_REPO_ROOT = Path(__file__).resolve().parents[2]

ContentInputT = Literal["checkbox", "radio"]


InputTypeT = Literal["IMAGE", "TEXT"]


ModelFrameworkT = Literal["pytorch", "tensorflow"]


ModelNameT = Literal[
    "bert-base-multilingual-cased",
    "distilbert-base-uncased",
    "distilbert-base-cased",
    "efficientnet_b0",
    "resnet50",
    "ultralytics/yolov5",
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
]


ModelRepositoryT = Literal["huggingface", "ultralytics", "torchvision", "detectron2"]


MLTaskT = Literal["CLASSIFICATION", "NAMED_ENTITIES_RECOGNITION", "OBJECT_DETECTION"]


ToolT = Literal["rectangle", "semantic"]


class Job(TypedDict):
    content: Dict  # type: ignore
    ml_task: MLTaskT
    tools: List[ToolT]
