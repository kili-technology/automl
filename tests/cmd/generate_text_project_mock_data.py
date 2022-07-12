import json
import os

import requests
from kili.client import Kili

if __name__ == "__main__":

    kili = Kili(api_key=os.environ["KILI_API_KEY"])
    fields = [
        "id",
        "externalId",
        "content",
        "labels.createdAt",
        "labels.jsonResponse",
        "labels.labelType",
    ]

    project_id = "cl1e4umogdgon0ly4737z82lc"
    g = kili.assets(
        project_id=project_id,
        fields=fields,
        status_in=["LABELED", "TO_REVIEW", "REVIEWED"],
        as_generator=True,
    )

    assets = list(g)[:52]

    with open("tests/e2e/fixtures/text_assets_fixture.json", "w") as f:
        json.dump(assets, f)

    c = {}
    api_key = os.environ["KILI_API_KEY"]
    for a in assets:
        response = requests.get(
            a["content"],  # type: ignore
            headers={
                "Authorization": f"X-API-Key: {api_key}",
                "PROJECT_ID": project_id,
            },
        )
        assert response.status_code == 200
        text = response.text
        c[a["content"]] = text  # type:ignore

    with open("tests/e2e/fixtures/text_content_fixture.json", "w") as f:
        json.dump(c, f)

    project = kili.projects(project_id=project_id)  # type:ignore

    del project[0]["roles"]  # type:ignore
    with open("tests/e2e/fixtures/text_project_fixture.json", "w") as f:
        json.dump(project, f)
