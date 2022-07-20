from click.testing import CliRunner

import main
from tests.e2e.utils_test_e2e import (
    debug_subprocess_pytest,
    mock__get_asset_memoized,
    mock__projects,
)


def mocked__download_asset_binary(api_key, asset_content):
    _ = api_key
    import pickle

    id = asset_content.split("label/v2/files?id=")[-1]
    with open(f"tests/e2e/fixtures/download_asset_binary/object_detection/{id}.pkl", "rb") as f:
        asset_data = pickle.load(f)
    return asset_data


def test_object_detection(mocker):

    mocker.patch("kili.client.Kili.__init__", return_value=None)
    mocker.patch(
        "kili.client.Kili.projects",
        side_effect=mock__projects("tests/e2e/fixtures/object_detection_project_fixture.json"),
    )
    mocker.patch(
        "kiliautoml.utils.download_assets.download_asset_binary",
        side_effect=mocked__download_asset_binary,
    )
    mocker.patch(
        "kiliautoml.utils.helpers.get_asset_memoized",
        side_effect=mock__get_asset_memoized(
            "tests/e2e/fixtures/object_detection_assets_fixture.json"
        ),
    )
    mocker.patch("commands.label_errors.upload_errors_to_kili")
    mocker.patch("kili.client.Kili.create_predictions")

    runner = CliRunner()
    project_id = "abcdefg"
    result = runner.invoke(
        main.kiliautoml,
        [
            "train",
            "--project-id",
            project_id,
            "--max-assets",
            "10",
            "--disable-wandb",
            "--epochs",
            "1",
        ],
    )
    debug_subprocess_pytest(result)

    result = runner.invoke(
        main.kiliautoml,
        [
            "predict",
            "--project-id",
            project_id,
            "--max-assets",
            "10",
        ],
    )
    debug_subprocess_pytest(result)
