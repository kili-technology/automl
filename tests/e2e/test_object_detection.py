import json

from click.testing import CliRunner

import predict
import train
from tests.e2e.utils_test_e2e import debug_subprocess_pytest


def mocked__get_assets(*_, max_assets=None, labeling_statuses=None):
    return json.load(open("tests/e2e/fixtures/object_detection_assets_fixture.json"))[:max_assets]


def mocked__projects(*_, project_id, fields):
    return json.load(open("tests/e2e/fixtures/object_detection_project_fixture.json"))


def mocked__download_asset_binary(api_key, asset_content, project_id):
    import pickle

    id = asset_content.split("label/v2/files?id=")[-1]
    with open(f"tests/e2e/fixtures/download_asset_binary/object_detection/{id}.pkl", "rb") as f:
        asset_data = pickle.load(f)
    return asset_data


def test_object_detection(mocker):

    mocker.patch("kili.client.Kili.__init__", return_value=None)
    mocker.patch("kili.client.Kili.projects", side_effect=mocked__projects)
    mocker.patch(
        "kiliautoml.utils.download_assets.download_asset_binary",
        side_effect=mocked__download_asset_binary,
    )
    mocker.patch("train.get_assets", side_effect=mocked__get_assets)
    mocker.patch("predict.get_assets", side_effect=mocked__get_assets)
    mocker.patch("label_errors.get_assets", side_effect=mocked__get_assets)
    mocker.patch("label_errors.upload_errors_to_kili")
    mocker.patch("kili.client.Kili.create_predictions")

    runner = CliRunner()
    result = runner.invoke(
        train.main,
        [
            "--api-endpoint",
            "https://staging.cloud.kili-technology.com/api/label/v2/graphql",
            "--project-id",
            "cl0wihlop3rwc0mtj9np28ti2",
            "--max-assets",
            "300",
            "--disable-wandb",
            "--epochs",
            "1",
        ],
    )
    debug_subprocess_pytest(result)

    result = runner.invoke(
        predict.main,
        [
            "--api-endpoint",
            "https://staging.cloud.kili-technology.com/api/label/v2/graphql",
            "--project-id",
            "cl0wihlop3rwc0mtj9np28ti2",
            "--max-assets",
            "300",
        ],
    )
    debug_subprocess_pytest(result)
