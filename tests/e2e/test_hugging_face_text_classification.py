from click.testing import CliRunner

import main
from tests.e2e.utils_test_e2e import (
    debug_subprocess_pytest,
    mock__get_asset_memoized,
    mock__projects,
    mocked__get_text_from,
)


def test_hugging_face_text_classification(mocker):
    mocker.patch("kili.client.Kili.__init__", return_value=None)
    mocker.patch(
        "kiliautoml.utils.helpers.get_asset_memoized",
        side_effect=mock__get_asset_memoized("tests/e2e/fixtures/text_assets_fixture.json"),
    )
    mocker.patch(
        "kili.client.Kili.projects",
        side_effect=mock__projects("tests/e2e/fixtures/text_project_fixture.json"),
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
