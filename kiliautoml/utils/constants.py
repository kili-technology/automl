import os

from typing_extensions import Literal

HOME = os.path.join(os.getenv("HOME"), ".cache", "kili", "automl")  # type: ignore


class ContentInput:
    Checkbox = "checkbox"
    Radio = "radio"


ContentInputT = Literal["checkbox", "radio"]


class InputType:
    Image = "IMAGE"
    Text = "TEXT"


InputTypeT = Literal["IMAGE", "TEXT"]


class ModelFramework:
    PyTorch = "pytorch"
    Tensorflow = "tensorflow"


ModelFrameworkT = Literal["pytorch", "tensorflow"]


class ModelName:
    EfficientNetB0 = "efficientnet_b0"
    Resnet50 = "resnet50"
    YoloV5 = "ultralytics/yolov5"


ModelNameT = Literal[
    "bert-base-multilingual-cased",
    "distilbert-base-cased",
    "efficientnet_b0",
    "resnet50",
    "ultralytics/yolov5",
]


class ModelRepository:
    HuggingFace = "huggingface"
    Ultralytics = "ultralytics"
    TorchVision = "torchvision"


ModelRepositoryT = Literal["huggingface", "ultralytics", "torchvision"]


class MLTask:
    Classification = "CLASSIFICATION"
    NamedEntityRecognition = "NAMED_ENTITIES_RECOGNITION"
    ObjectDetection = "OBJECT_DETECTION"


MLTaskT = Literal["CLASSIFICATION", "NAMED_ENTITIES_RECOGNITION", "OBJECT_DETECTION"]


class Tool:
    Rectangle = "rectangle"


ToolT = Literal["rectangle"]
