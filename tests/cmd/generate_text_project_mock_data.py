import json
import os
import sys

import requests
from kili.client import Kili

if __name__ == "__main__":
    project_id = sys.argv[1]

    kili = Kili()
    g = kili.assets(project_id=project_id, as_generator=True)

    assets = []
    for i, a in enumerate(g):
        assets.append(a)
        if i > 50:
            break

    for a in assets:
        for label in a["labels"]:
            del label["author"]

    with open("tests/e2e/fixtures/text_assets_fixture.json", "w") as f:
        json.dump(assets, f)

    project = kili.projects(project_id=project_id)[0]  # type:ignore

    del project["roles"]
    with open("tests/e2e/fixtures/text_project_fixture.json", "w") as f:
        json.dump(project, f)

    c = {}
    api_key = os.environ["KILI_API_KEY"]
    for a in assets:
        response = requests.get(
            a["content"],
            headers={
                "Authorization": f"X-API-Key: {api_key}",
                "PROJECT_ID": project_id,
            },
        )
        assert response.status_code == 200
        text = response.text
        c[a["content"]] = text

    with open("tests/e2e/fixtures/text_content_fixture.json", "w") as f:
        json.dump(c, f)
