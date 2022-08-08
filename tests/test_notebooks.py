"""
Test notebooks with pytest
"""

import os

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

RECIPES_DIR = "./notebooks"
RECIPES_TESTED = [
    "image_classification.ipynb",
    "named_entity_recognition.ipynb",
    "object_detection.ipynb",
    "semantic_segmentation.ipynb",
    "text_classification.ipynb",
]


def process_notebook(notebook_filename):
    """
    Checks if an IPython notebook runs without error from start to finish.
    """
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=1000, kernel_name="python3")

    try:
        # Check that the notebook runs
        ep.preprocess(nb, {"metadata": {"path": ""}})
    except CellExecutionError:
        raise

    print(f"Successfully executed {notebook_filename}")
    return


def test_all_recipes():
    """
    Runs `process_notebook` on all notebooks in the git repository.
    """
    notebooks = [os.path.join(RECIPES_DIR, recipe) for recipe in RECIPES_TESTED]
    for notebook in notebooks:
        print("Testing", notebook)
        process_notebook(notebook)

    return


if __name__ == "__main__":
    test_all_recipes()
