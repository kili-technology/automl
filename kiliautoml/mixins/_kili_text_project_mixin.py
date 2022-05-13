from abc import ABCMeta

from kili.client import Kili

from kiliautoml.utils.download_assets import download_asset_unicode
from kiliautoml.utils.type import AssetT


class KiliTextProjectMixin(metaclass=ABCMeta):
    def __init__(self, project_id: str, api_key: str, api_endpoint: str) -> None:
        self.project_id = project_id
        self.api_key = api_key

        self.kili = Kili(api_key=api_key, api_endpoint=api_endpoint)

    def _get_text_from(self, asset_url: str) -> str:
        text = download_asset_unicode(self.api_key, asset_url)
        return text
