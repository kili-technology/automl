from click.testing import CliRunner

from tests.e2e.utils_test_e2e import one_command, prepare_mocker

MOCK_DIR = "cl668phvq00220lyyhpv38myk_ner"
project_id = MOCK_DIR.split("_")[0]


def test_ner(mocker):
    prepare_mocker(mocker, MOCK_DIR)

    runner = CliRunner()
    one_command(runner, "advise", project_id)
    one_command(runner, "train", project_id)
    one_command(runner, "predict", project_id)
