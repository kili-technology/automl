from dataclasses import dataclass
import os


HOME = os.path.join(os.getenv("HOME"), ".cache", "kili", "automl")  # type: ignore


@dataclass
class ContentInput:
    Checkbox = "checkbox"
    Radio = "radio"


@dataclass
class InputType:
    Image = "IMAGE"
    Text = "TEXT"


@dataclass
class ModelFramework:
    PyTorch = "pytorch"
    Tensorflow = "tensorflow"


@dataclass
class ModelName:
    BertBaseMultilingualCased = "bert-base-multilingual-cased"
    DistilbertBaseCased = "distilbert-base-cased"
    EfficientNetB0 = "efficientnet_b0"
    Resnet50 = "resnet50"
    YoloV5 = "ultralytics/yolov5"


@dataclass
class ModelRepository:
    HuggingFace = "huggingface"
    Ultralytics = "ultralytics"


@dataclass
class MLTask:
    Classification = "CLASSIFICATION"
    NamedEntityRecognition = "NAMED_ENTITIES_RECOGNITION"
    ObjectDetection = "OBJECT_DETECTION"


@dataclass
class Tool:
    Rectangle = "rectangle"
