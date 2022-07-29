import json

from click.testing import CliRunner

import main
from tests.e2e.utils_test_e2e import create_mock__projects, debug_subprocess_pytest

text_content = json.load(open("tests/e2e/fixtures/text_content_fixture.json"))


def mocked__get_text_from(asset_url):
    return text_content[asset_url]


def mocked__get_asset_memoized(**kwargs):
    res = json.load(open("tests/e2e/fixtures/text_assets_fixture.json"))
    max_assets = kwargs.get("max_assets", None)
    tot = min(20, max_assets) if max_assets is not None else 20

    project_id = "abcdefgh"
    if project_id == "abcdefgh":
        return res[:tot]
    elif project_id == "abcdefgh2":
        return res[-tot:]


def test_hugging_face_text_classification(mocker):

    mocker.patch("kili.client.Kili.__init__", return_value=None)
    mocker.patch(
        "kiliautoml.utils.helpers.get_asset_memoized", side_effect=mocked__get_asset_memoized
    )
    mocker.patch(
        "kili.client.Kili.projects",
        side_effect=create_mock__projects("tests/e2e/fixtures/text_project_fixture.json"),
    )
    mocker.patch(
        "kiliautoml.mixins._kili_text_project_mixin.KiliTextProjectMixin._get_text_from",
        side_effect=mocked__get_text_from,
    )
    mock_create_predictions = mocker.patch("kili.client.Kili.create_predictions")

    runner = CliRunner()
    result = runner.invoke(
        main.kiliautoml,
        [
            "train",
            "--project-id",
            "abcdefgh",
            "--max-assets",
            "20",
            "--target-job",
            "CLASSIFICATION_JOB_0",
            "--model-name",
            "distilbert-base-cased",
            "--disable-wandb",
            "--clear-dataset-cache",
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ],
    )
    debug_subprocess_pytest(result)

    result = runner.invoke(
        main.kiliautoml,
        [
            "predict",
            "--project-id",
            "abcdefgh2",
            "--max-assets",
            "10",
            "--target-job",
            "CLASSIFICATION_JOB_0",
            "--project-id",
            "abcdefgh",
            "--batch-size",
            "2",
        ],
    )
    debug_subprocess_pytest(result)
    assert result.output.count("OPTIMISM") == 0
    mock_create_predictions.assert_called_once()
