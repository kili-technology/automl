import os

from typing_extensions import Literal

AUTOML_CACHE = os.getenv(
    "KILIAUTOML_CACHE", os.path.join(os.getenv("HOME"), ".cache", "kili", "automl")  # type:ignore
)

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
