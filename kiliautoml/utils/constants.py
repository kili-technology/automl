import os
from typing import Dict, List

from typing_extensions import Literal, TypedDict

HOME = os.path.join(os.getenv("HOME"), ".cache", "kili", "automl")  # type: ignore


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
]


ModelRepositoryT = Literal["huggingface", "ultralytics", "torchvision"]


MLTaskT = Literal["CLASSIFICATION", "NAMED_ENTITIES_RECOGNITION", "OBJECT_DETECTION"]


ToolT = Literal["rectangle"]


class Job(TypedDict):
    content: Dict  # type: ignore
    ml_task: MLTaskT
    tools: List[ToolT]
