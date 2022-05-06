from abc import ABCMeta
from typing import Dict

from kili.client import Kili

from kiliautoml.utils.download_assets import download_asset_unicode


class KiliTextProjectMixin(metaclass=ABCMeta):
    def __init__(self, project_id: str, api_key: str, api_endpoint: str) -> None:
        self.project_id = project_id
        self.api_key = api_key

        self.kili = Kili(api_key=api_key, api_endpoint=api_endpoint)

    def _get_text_from(self, asset: Dict) -> str:
        text = download_asset_unicode(self.api_key, asset["content"], self.project_id)
        return text
