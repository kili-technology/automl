from abc import ABCMeta
from typing import Dict

import requests
from kili.client import Kili


class KiliTextProjectMixin(metaclass=ABCMeta):
    def __init__(self, project_id: str, api_key: str, api_endpoint: str) -> None:
        self.project_id = project_id
        self.api_key = api_key

        self.kili = Kili(api_key=api_key, api_endpoint=api_endpoint)

    def _get_text_from(self, asset: Dict) -> str:
        response = requests.get(
            asset["content"],  # type:ignore
            headers={
                "Authorization": f"X-API-Key: {self.api_key}",
            },
        )

        text = response.text
        return text
