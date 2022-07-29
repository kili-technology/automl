from click.testing import CliRunner

import main
from kiliautoml.utils.type import AssetT
from tests.e2e.utils_test_e2e import (
    debug_subprocess_pytest,
    mock__get_asset_memoized,
    mock__projects,
)

MOCK_DIR = "cl56hzgbp0ix60lst97r56err_segmentation"


def mocked__throttled_request(api_key, asset_content):
    _ = api_key
    function_name = "throttled_request"
    path = f"tests/e2e/fixtures/{MOCK_DIR}/{function_name}"
    import pickle

    id = asset_content.split("/")[-1].split(".")[0]
    with open(f"{path}/{id}.pkl", "rb") as f:
        asset_data = pickle.load(f)
    return asset_data


def mocked_iter_refreshed_asset(kili):
    _ = kili
    assets = mock__get_asset_memoized(f"tests/e2e/fixtures/{MOCK_DIR}/assets.json")()
    for asset in assets:
        yield AssetT.construct(**asset)


def test_detectron2_image_segmentation(mocker):

    mocker.patch("kili.client.Kili.__init__", return_value=None)
    mocker.patch(
        "kiliautoml.utils.type.AssetsLazyList.iter_refreshed_asset",
        side_effect=mocked_iter_refreshed_asset,
    )
    mocker.patch(
        "kili.client.Kili.projects",
        side_effect=mock__projects(f"tests/e2e/fixtures/{MOCK_DIR}/projects.json"),
    )
    mocker.patch(
        "kiliautoml.utils.helpers.get_asset_memoized",
        side_effect=mock__get_asset_memoized(f"tests/e2e/fixtures/{MOCK_DIR}/assets.json"),
    )
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
