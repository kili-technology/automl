"""To generate new mocked data, just launch kiliautoml like this:

0. Create a new projet with 15 assets and label everyone of them.
1. USE THIS COMMAND:
export KILI_AUTOML_MOCK=True  && export PROJECTID=cl66k1tvd9bhd0lz94q1x852l \
    && export KILI_AUTOML_MOCK_OUTPUT_DIR=cl66k1tvd9bhd0lz94q1x852l_text_classification \
    && kiliautoml train \
        --project-id $PROJECTID --clear-dataset-cache --epochs 1 \
    && export KILI_AUTOML_MOCK=

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
    id = strip_token(id)
    dir_path = f"tests/e2e/fixtures/{MOCK_DIR}/{function_name}"
    os.makedirs(dir_path, exist_ok=True)

    pickle.dump(response, open(f"{dir_path}/{id}.pkl", "wb"))
    print("Saved ", id)


def strip_token(id):
    if "&token=" in id:
        id = id.split("&token=")[0].split("files?id=")[1]
    return id


def jsonify_mock_data(res, function_name):
    dir_path = f"tests/e2e/fixtures/{MOCK_DIR}"
    os.makedirs(dir_path, exist_ok=True)

    with open(f"{dir_path}/{function_name}.json", "w") as f:
        json.dump(res, f)
