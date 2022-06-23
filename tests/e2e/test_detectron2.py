import json

from click.testing import CliRunner

import main
from tests.e2e.utils_test_e2e import debug_subprocess_pytest

MOCK_DIR = "cl4cisaq36awx0lpb8ql57mxk_segmentation"


def mocked__get_assets(*_, max_assets=None, randomize=None):
    _ = randomize
    function_name = "assets"
    path = f"tests/e2e/fixtures/{MOCK_DIR}/{function_name}.json"
    return json.load(open(path))[:max_assets]


def mocked__projects(*_, project_id, fields):
    _ = project_id, fields

    function_name = "projects"
    path = f"tests/e2e/fixtures/{MOCK_DIR}/{function_name}.json"
    return json.load(open(path))


def mocked__throttled_request(api_key, asset_content):
    _ = api_key
    function_name = "throttled_request"
    path = f"tests/e2e/fixtures/{MOCK_DIR}/{function_name}"
    import pickle

    id = asset_content.split("/")[-1].split(".")[0]
    with open(f"{path}/{id}.pkl", "rb") as f:
        asset_data = pickle.load(f)
    return asset_data


def test_detectron2_image_segmentation(mocker):

    mocker.patch("kili.client.Kili.__init__", return_value=None)
    mocker.patch("kili.client.Kili.projects", side_effect=mocked__projects)
    mocker.patch("kiliautoml.utils.helpers.get_assets", side_effect=mocked__get_assets)
    mocker.patch(
        "kiliautoml.utils.download_assets.throttled_request",
        side_effect=mocked__throttled_request,
    )

    runner = CliRunner()
    result = runner.invoke(
        main.kiliautoml,
        [
            "train",
            "--project-id",
            "cl4cisaq36awx0lpb8ql57mxk",
            "--target-job",
            "JOB_0",
            "--disable-wandb",
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ],
    )
    debug_subprocess_pytest(result)

    mock_create_predictions = mocker.patch("kili.client.Kili.create_predictions")
    mocker.patch("commands.predict.get_assets", side_effect=mocked__get_assets)
    result = runner.invoke(
        main.kiliautoml,
        [
            "predict",
            "--project-id",
            "cl4cisaq36awx0lpb8ql57mxk",
            "--target-job",
            "JOB_0",
            "--batch-size",
            "2",
            "--dry-run",
        ],
    )
    debug_subprocess_pytest(result)
    mock_create_predictions.assert_not_called()
