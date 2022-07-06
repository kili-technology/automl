"""To generate new mocked data, just launch kiliautoml like this:

1. USE THIS COMMAND:
KILI_AUTOML_MOCK=True KILI_AUTOML_MOCK_OUTPUT_DIR=cl56hzgbp0ix60lst97r56err_segmentation \
    kiliautoml train \
        --clear-dataset-cache --epochs 1 \
        --project-id cl56hzgbp0ix60lst97r56err

2. ADAPT THE TEST FILE
"""
import json
import os
import pickle

GENERATE_MOCK = os.getenv("KILI_AUTOML_MOCK", False)

# Example : "cl56hzgbp0ix60lst97r56err_segmentation"
MOCK_DIR = os.getenv("KILI_AUTOML_MOCK_OUTPUT_DIR", None)

if GENERATE_MOCK:
    print("We will generate the mocking test data.")


def save_mock_data(id, response, function_name):
    dir_path = f"tests/e2e/fixtures/{MOCK_DIR}/{function_name}"
    os.makedirs(dir_path, exist_ok=True)

    pickle.dump(response, open(f"{dir_path}/{id}.pkl", "wb"))
    print("Saved ", id)


def jsonify_mock_data(res, function_name):
    dir_path = f"tests/e2e/fixtures/{MOCK_DIR}"
    os.makedirs(dir_path, exist_ok=True)

    with open(f"{dir_path}/{function_name}.json", "w") as f:
        json.dump(res, f)
