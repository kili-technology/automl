import json
from typing import Any, List

from pytest_mock import MockerFixture

from kiliautoml.utils.type import AssetT, CommandT

text_content = json.load(open("tests/e2e/fixtures/text_content_fixture.json"))


def mocked__get_text_from(asset_url):
    return text_content[asset_url]


def mock__get_asset_memoized(path):
    def mocked__get_asset_memoized(**kwargs) -> List[Any]:
        total = kwargs.get("total", None)
        loaded_assets = json.load(open(path))
        if total is not None:
            return loaded_assets[:total]
        else:
            return loaded_assets

    return mocked__get_asset_memoized


def mock__projects(path):
    def mocked__projects(*_, project_id, fields):
        _ = project_id, fields
        return json.load(open(path))

    return mocked__projects


def debug_subprocess_pytest(result):
    import traceback

    print(result.output)
    if result.exception is not None:
        tb = result.exception.__traceback__
        traceback.print_tb(tb)
        raise Exception(result.exception)
    assert result.exit_code == 0


def create_mocked__throttled_request(path_dir):
    def mocked__throttled_request(api_key, asset_content):
        _ = api_key
        function_name = "throttled_request"
        path = f"{path_dir}/{function_name}"
        import pickle

        id = asset_content.split("/")[-1].split(".")[0]
        with open(f"{path}/{id}.pkl", "rb") as f:
            asset_data = pickle.load(f)
        return asset_data

    return mocked__throttled_request


def create_mocked_iter_refreshed_asset(path):
    def mocked_iter_refreshed_asset(kili):
        _ = kili
        assets = mock__get_asset_memoized(path)()
        for asset in assets:
            yield AssetT.construct(**asset)

    return mocked_iter_refreshed_asset


def prepare_mocker(mocker: MockerFixture, MOCK_DIR: str):
    mocker.patch("kili.client.Kili.__init__", return_value=None)
    mocker.patch(
        "kiliautoml.utils.type.AssetsLazyList.iter_refreshed_asset",
        side_effect=create_mocked_iter_refreshed_asset(
            f"tests/e2e/fixtures/{MOCK_DIR}/assets.json"
        ),
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
        side_effect=create_mocked__throttled_request(f"tests/e2e/fixtures/{MOCK_DIR}"),
    )


def create_arguments_test(command: CommandT, project_id, target_job="JOB_0"):
    if command == "train":
        args = [
            command,
            "--project-id",
            project_id,
            "--target-job",
            target_job,
            "--disable-wandb",
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ]
    else:
        args = [
            command,
            "--project-id",
            project_id,
            "--target-job",
            target_job,
            "--batch-size",
            "2",
            "--dry-run",
        ]
    return args
