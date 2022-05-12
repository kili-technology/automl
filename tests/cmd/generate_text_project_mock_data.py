import json
import os
import sys

import requests
from kili.client import Kili

if __name__ == "__main__":
    project_id = sys.argv[1]

    kili = Kili(api_key=os.environ["KILI_API_KEY"])
    g = kili.assets(project_id=project_id, as_generator=True)

    assets = list(g)[:50]

    for a in assets:
        for label in a["labels"]:
            del label["author"]

    with open("tests/e2e/fixtures/object_detection_assets_fixture.json", "w") as f:
        json.dump(assets, f)

    project = kili.projects(project_id=project_id)  # type:ignore

    del project[0]["roles"]
    with open("tests/e2e/fixtures/object_detection_project_fixture.json", "w") as f:
        json.dump(project, f)

    c = {}
    api_key = os.environ["KILI_API_KEY"]
    for a in assets:
        response = requests.get(
            a["content"],  # type:ignore
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
