from click.testing import CliRunner

from tests.e2e.utils_test_e2e import one_command, prepare_mocker

MOCK_DIR = "cl66c5zr97wdw0mww1oz88etx_yolo"
project_id = MOCK_DIR.split("_")[0]


def test(mocker):

    prepare_mocker(mocker, MOCK_DIR)

    runner = CliRunner()
    one_command(runner, "advise", project_id)
    one_command(runner, "train", project_id)
    one_command(runner, "predict", project_id)
