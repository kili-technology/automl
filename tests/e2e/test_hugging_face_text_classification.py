import json

from click.testing import CliRunner

import main
from tests.e2e.utils_test_e2e import debug_subprocess_pytest

text_content = json.load(open("tests/e2e/fixtures/text_content_fixture.json"))


def mocked__get_text_from(asset_url):
    return text_content[asset_url]


def mocked__get_assets(*_, max_assets=None, randomize=None):
    _ = randomize
    return json.load(open("tests/e2e/fixtures/text_assets_fixture.json"))[:max_assets]


def mocked__projects(*_, project_id, fields):
    _ = project_id, fields
    return json.load(open("tests/e2e/fixtures/text_project_fixture.json"))


def test_hugging_face_text_classification(mocker):

    mocker.patch("kili.client.Kili.__init__", return_value=None)
    mocker.patch("kiliautoml.utils.helpers.get_assets", side_effect=mocked__get_assets)
    mocker.patch("kili.client.Kili.projects", side_effect=mocked__projects)
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
            "4",
            "--target-job",
            "CLASSIFICATION_JOB_0",
            "--model-name",
            "distilbert-base-uncased",
            "--disable-wandb",
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ],
    )
    debug_subprocess_pytest(result)

    mocker.patch("commands.predict.get_assets", side_effect=mocked__get_assets)

    result = runner.invoke(
        main.kiliautoml,
        [
            "predict",
            "--project-id",
            "abcdefgh",
            "--max-assets",
            "10",
            "--target-job",
            "CLASSIFICATION_JOB_0",
            "--batch-size",
            "2",
        ],
    )
    debug_subprocess_pytest(result)
    assert result.output.count("OPTIMISM") == 0
    mock_create_predictions.assert_called_once()
    # Note: useful for debugging:
    # import traceback
    # print(traceback.print_tb(result.exception.__traceback__))

    mock_create_predictions = mocker.patch("kili.client.Kili.create_predictions")

    result = runner.invoke(
        main.kiliautoml,
        [
            "predict",
            "--project-id",
            "abcdefgh",
            "--max-assets",
            "10",
            "--target-job",
            "CLASSIFICATION_JOB_0",
            "--dry-run",
            "--verbose",
            "1",
            "--batch-size",
            "2",
        ],
    )
    debug_subprocess_pytest(result)
    words = ["OPTIMISM", "ENTHUSIASM", "CONCERN", "ANGER", "FEAR", "UNCERTAIN"]
    assert sum(result.output.count(c) for c in words) == 10

    mock_create_predictions.assert_not_called()
