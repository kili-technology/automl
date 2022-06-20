"""To generate new mocked data, just launch kiliautoml train with the --clean-dataset-cache

and change GENERATE_MOCK to True.
and choose a MOCK_DIR.
"""
import json
import os
import pickle

GENERATE_MOCK = False
MOCK_DIR = "cl4cisaq36awx0lpb8ql57mxk_segmentation"


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
