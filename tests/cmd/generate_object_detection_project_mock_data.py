import json
import os

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

    project_id = "cl0wihlop3rwc0mtj9np28ti2"

    g = kili.assets(
        project_id=project_id,
        fields=fields,
        status_in=["LABELED", "TO_REVIEW", "REVIEWED"],
        as_generator=True,
    )
    assets = list(g)[:50]

    with open("tests/e2e/fixtures/object_detection_assets_fixture.json", "w") as f:
        json.dump(assets, f)

    project = kili.projects(project_id=project_id)  # type:ignore

    del project[0]["roles"]  # type:ignore
    with open("tests/e2e/fixtures/object_detection_project_fixture.json", "w") as f:
        json.dump(project, f)
