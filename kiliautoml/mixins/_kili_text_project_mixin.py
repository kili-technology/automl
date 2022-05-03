from abc import ABCMeta

import requests
from kili.client import Kili


class KiliTextProjectMixin(metaclass=ABCMeta):
    def __init__(self, project_id: str, api_key: str, api_endpoint: str) -> None:
        self.project_id = project_id
        self.api_key = api_key

        self.kili = Kili(api_key=api_key, api_endpoint=api_endpoint)

    def _get_text_from(self, asset_url: str) -> str:
        response = requests.get(
            asset_url,  # type:ignore
            headers={
                "Authorization": f"X-API-Key: {self.api_key}",
            },
        )
        assert response.status_code == 200
        text = response.text
        return text
