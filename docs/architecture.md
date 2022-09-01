
# Architecture

## KiliModels

- kiliautoml/models/_base_model.py : Gives an example of a Parent Model class.
- kiliautoml/models/_detectron2_semantic_segmentation.py : Gives an example of model (Child class)
- kiliautoml/models/kili_auto_model.py : Only public interface. This class encapsulates all the other models, but be careful, it does not exactly match the model of the parent classin _base_model.py

## The command file

- When using the line command, the instruction is redirected to the file corresponding to the requested command.
- The documentation and the default parameters of each command are located in the commands/common_args.py file
- You can read the click librairy documentation, section https://click.palletsprojects.com/en/8.1.x/setuptools/

## The Helper file

The helper file is a file with little structure, but allowing to factorize in the same file functions reused several times in different other files.
You will find utilities to download asets, to download projects, and other functions with a more rare use.


## The Typing file

Since Python 3.6, it is possible to use type hint in Python. We use Pylance and Mypy to check the consistency of types. We have finely tuned Pylance to suit our needs.
We type AutoML assuming that the executed version is python 3.7, ie, the version of python used by google Colab. When using advanced type objects, we import them from the typing_extension library, which allows us to be backward compatible and use python 3.7 up to 3.10.


We took a particular care to type hint the AutoML library in a rigorous way. This makes the development experience more pleasant and increase the speed of iteration considerably.

But we use a relatively complicated type of system:
- The data structures are either in the form of TypedDict, DataClass or pydantic model.
    - TypedDict allows to use python dictionaries without overhead, but with the benefit of Pylance type checking. The Kili SDK uses dictionaries, so it is the default solution to type json for assets.
    - DataClass is used when we don't work directly on assets. Dataclasses are typically obkects specific to autoML, but which don't exist in the SDK documentaiton.
    - Pydantic is a type verification library. It is mainly used when converting typedDict representing annotations into standardized annotations. Before standardizing the annotations, we proceed to verifications to check that the labels have the right format (example: a bbox must contain 4 points, a polygon at least 3 points, etc...)
- You can check the list of Pylance parameters in the pyproject.toml. For exemple,
  - reportIncompatibleMethodOverride is very important to ensure that the child classes are consistent with the Parent class.
  - reportUnusedVariable is important to ensure that all paramters in a function are used. Is a parameter is not used you need to use something like '_ = unused_parameter'.


The types are then distributed to all other files in the application.
