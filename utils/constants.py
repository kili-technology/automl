from dataclasses import dataclass
import os


HOME = os.path.join(os.getenv('HOME'), '.cache', 'kili', 'automl')

@dataclass
class ContentInput:
    Checkbox = 'checkbox'
    Radio = 'radio'

@dataclass
class InputType:
    Image = 'IMAGE'
    Text = 'TEXT'

@dataclass
class ModelFramework:
    PyTorch = 'pytorch'
    Tensorflow = 'tensorflow'

@dataclass
class ModelName:
    BertBaseMultilingualCased = 'bert-base-multilingual-cased'
    

@dataclass
class ModelRepository:
    HuggingFace = 'huggingface'
    Yolo = 'yolo'

@dataclass
class MLTask:
    Classification = 'CLASSIFICATION'
    NamedEntitiesRecognition = 'NAMED_ENTITIES_RECOGNITION'
    ObjectDetection = 'OBJECT_DETECTION'

@dataclass
class Tool:
    Rectangle = 'rectangle'