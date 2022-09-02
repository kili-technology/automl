
# Architecture

## KiliModels

- [kiliautoml/models/_base_model.py] : Gives an example of a parent Model class.
- [kiliautoml/models/_detectron2_semantic_segmentation.py] : Gives an example of model (Child class)
- [kiliautoml/models/kili_auto_model.py] : Only public interface. This class encapsulates all the other models, but be careful, it does not exactly match the model of the parent class in [_base_model.py]

## The command file

- When using the line command, the instruction is redirected to the file corresponding to the requested command.
- The documentation and the default parameters of each command are located in the commands/common_args.py file
- You can read the click library documentation, section https://click.palletsprojects.com/en/8.1.x/setuptools/

## The Helper file

The helper file is a file with little structure, but allowing to factorize in the same file functions reused several times in different other files.
You will find utilities to download asets, to download projects, and other functions more rarely used.


## The Typing file

Since Python 3.6, it is possible to use type hints in Python. We use Pylance and Mypy to check the consistency of types. We have finely tuned Pylance to suit our needs.
We type AutoML assuming that the executed version is python 3.7, ie, the version of python used by google Colab. When using advanced type objects, we import them from the typing_extension library, which allows us to be backward compatible and to use python 3.7 up to 3.10.


We took a particular care to type hint the AutoML library in a rigorous way. This makes the development experience more pleasant and increase the speed of iteration considerably.

But we use a relatively complicated system of type:
- The data structures are either in the form of TypedDict, DataClass or pydantic model.
    - TypedDict allows to use python dictionaries without overhead, but with the benefit of Pylance type checking. The Kili SDK uses dictionaries, so it is the default solution to type json containing assets.
    - DataClass is used when we don't work directly on assets. Dataclasses are typically objects specific to autoML, but which don't exist in the SDK documentation.
    - Pydantic is a type verification library. It is mainly used when converting typedDict representing annotations into standardized annotations (The standardized annotations is a way to represent any annotation from any modality to compute the label errors). Before standardizing the annotations, we proceed to check that the labels have the right formats (example: a bbox must contain 4 points, a polygon at least 3 points, etc...)
- You can check the list of Pylance parameters in the [pyproject.toml] file. For exemple,
  - `reportIncompatibleMethodOverride` is very important to ensure that the child classes are consistent with the Parent class.
  - `reportUnusedVariable` is important to ensure that all paramters in a function are used. If a parameter is not used you need to use something like `_ = unused_parameter`.


The types are then distributed to all other files in the application.

## Particularities of each modality

- Image classification: This is the only modality that benefits from the prioritization module. It uses plain pytorchvision in backend.
- Text classification: It uses hugggingface in backend.
- Image segmentation and polygon: Image segmentation and bbox are two different things in the Kili interface but share exactly the same code in AutoML. The difference between the two is not a qualitative difference but simply a quantitative one: image segmentation = polygon with many sides, polygon = polygon with relatively few sides. They use the Detectron library in backend.


## System used in the LabelError command

- Image classification: Image classification uses cleanlab in the backend to detect annotation errors. The paradigm used by cleanlab consists in training several models on different parts of the data, then comparing the predictions of each model.
- Object Detection: Polygon, Bbox, Image Segmentation: These three modalities use the same error detection system which is hand coded. Once a model has been trained, we compare the model predictions with the manual annotations.
- NER could benefit from the same system as the object detection modalities, but is not yet implemented. However, it would be relatively easy to implement in a few extra lines of code. (implementing the IUO metric for Ner would be sufficient)
