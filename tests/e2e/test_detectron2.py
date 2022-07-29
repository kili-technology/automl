from click.testing import CliRunner

import main
from tests.e2e.utils_test_e2e import (
    create_arguments_test,
    debug_subprocess_pytest,
    prepare_mocker,
)

MOCK_DIR = "cl56hzgbp0ix60lst97r56err_segmentation"
project_id = MOCK_DIR.split("_")[0]


def test_detectron2_image_segmentation(mocker):

    prepare_mocker(mocker, MOCK_DIR)

    runner = CliRunner()
    result = runner.invoke(
        main.kiliautoml,
        create_arguments_test("train", project_id),
    )
    debug_subprocess_pytest(result)

    mock_create_predictions = mocker.patch("kili.client.Kili.create_predictions")
    result = runner.invoke(
        main.kiliautoml,
        create_arguments_test("predict", project_id),
    )
    debug_subprocess_pytest(result)
    mock_create_predictions.assert_not_called()
