from abc import ABCMeta

from kiliautoml.utils.download_assets import download_asset_unicode


# TODO: Delete this file
class KiliTextProjectMixin(metaclass=ABCMeta):
    def __init__(self, api_key) -> None:
        self.api_key = api_key

    def _get_text_from(self, asset_url: str) -> str:
        text = download_asset_unicode(self.api_key, asset_url)
        return text
