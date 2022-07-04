import json


text_content = json.load(open("tests/e2e/fixtures/text_content_fixture.json"))


def mocked__get_text_from(asset_url):
    return text_content[asset_url]


def mock__get_asset_memoized(path):
    def mocked__get_asset_memoized(**_):
        return json.load(open(path))

    return mocked__get_asset_memoized


def mock__projects(path):
    def mocked__projects(*_, project_id, fields):
        _ = project_id, fields
        return json.load(open(path))

    return mocked__projects


def debug_subprocess_pytest(result):
    import traceback

    print("result.output")
    print(result.output)
    if result.exception is not None:
        traceback.print_tb(result.exception.__traceback__)
        print(result.exception)
    assert result.exit_code == 0
