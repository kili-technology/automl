import json
import os
import pickle

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

    project_id = "cl48g1ata000l0no14mir80c0"
    assets = kili.assets(
        project_id=project_id,
        fields=fields,
        status_in=["LABELED", "TO_REVIEW", "REVIEWED"],
        first=300,
    )

    with open("tests/e2e/fixtures/img_class_get_assets_fixture.json", "w") as f:
        json.dump(assets, f)

    api_key = os.environ["KILI_API_KEY"]
    for a in assets:
        response = requests.get(
            a["content"],  # type:ignore
            headers={
                "Authorization": f"X-API-Key: {api_key}",
                "PROJECT_ID": project_id,
            },
        )

        id = a["content"].split("label/v2/files?id=")[-1]  # type:ignore
        with open(
            "tests/e2e/fixtures/download_asset_binary/image_classification/" + id + ".pkl",
            "wb",
        ) as f1:
            pickle.dump(response.content, f1)

    project = kili.projects(project_id=project_id)  # type:ignore

    del project[0]["roles"]  # type:ignore
    with open("tests/e2e/fixtures/img_class_project_fixture.json", "w") as f:
        json.dump(project, f)
