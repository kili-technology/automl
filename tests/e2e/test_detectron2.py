from click.testing import CliRunner

from tests.e2e.utils_test_e2e import one_command, prepare_mocker

MOCK_DIR = "cl66ckxol06vt0pylc35ofip2_detectron"
project_id = MOCK_DIR.split("_")[0]


def test(mocker):

    prepare_mocker(mocker, MOCK_DIR)

    runner = CliRunner()
    one_command(runner, "train", project_id)
    one_command(runner, "predict", project_id)
