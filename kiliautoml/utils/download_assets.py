import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image
from PIL.Image import Image as PILImage
from ratelimit import limits, sleep_and_retry
from requests import Response
from tqdm.autonotebook import tqdm

from kiliautoml.utils.helper_mock import GENERATE_MOCK, save_mock_data
from kiliautoml.utils.helpers import kili_print
from kiliautoml.utils.memoization import kili_memoizer
from kiliautoml.utils.type import AssetExternalIdT, AssetIdT, AssetT


@dataclass
class DownloadedImage:
    id: AssetIdT
    externalId: AssetExternalIdT
    filepath: str

    def get_image(self) -> PILImage:
        return Image.open(self.filepath)


@dataclass
class DownloadedText:
    id: str
    externalId: str
    content: str


DELAY = 60 / 250  # 250 calls per minutes


@sleep_and_retry
@limits(calls=1, period=DELAY)
def _throttled_request(api_key, asset_content, use_header=True, k=0) -> Response:  # type: ignore
    if k == 20:
        raise Exception("Too many retries")
    if use_header:
        response = requests.get(
            asset_content,
            headers={
                "Authorization": f"X-API-Key: {api_key}",
            },
        )
    else:
        response = requests.get(asset_content)
    try:
        assert response.status_code == 200

        if GENERATE_MOCK:
            id = asset_content.split("/")[-1].split(".")[0]
            save_mock_data(id, response, function_name="throttled_request")
        return response
    except AssertionError as e:
        # Sometimes, the header breaks google bucket and just removing the header makes it work.
        _ = e
        return _throttled_request(api_key, asset_content, use_header=not use_header, k=k + 1)


@kili_memoizer
def _throttled_request_memoized(api_key, asset_content, asset_id):
    _ = asset_id
    return _throttled_request(api_key, asset_content)


def throttled_request(api_key, asset_content):
    """
    asset_content contains the id and the token.
    'https://cloud.kili-technology.com/api/label/v2/files?id=3a5aa0ca-328e-4f0f-bedd-4ffff27f796d&token=890cc70e2'

    But we want to memoize the asset even if the token changes.
    """
    if "files?id=" in asset_content and "&token=" in asset_content:
        asset_id = asset_content.split("?id=")[1].split("&token=")[0]
        print("_throttled_request_memoized")
        return _throttled_request_memoized(api_key, asset_content, asset_id)
    else:
        print("_throttled_request")
        return _throttled_request(api_key, asset_content)


def download_asset_binary(api_key, asset_content):
    response = throttled_request(api_key, asset_content)
    asset_data = response.content
    return asset_data


def download_asset_unicode(api_key, asset_content):
    response = throttled_request(api_key, asset_content)
    text = response.text
    return text


def download_image(api_key, asset_content):
    img_data = download_asset_binary(api_key, asset_content)

    image = Image.open(BytesIO(img_data))
    return image


def download_image_retry(api_key, asset: AssetT, n_try: int):
    while n_try < 20:
        try:
            img_data = download_asset_binary(api_key, asset.content)
            break
        except Exception:
            time.sleep(1)
            n_try += 1
    return img_data  # type:ignore


def download_project_images(
    api_key: str,
    assets: List[AssetT],
    output_folder: Optional[str] = None,
) -> List[DownloadedImage]:
    kili_print("Downloading images to folder {}".format(output_folder))
    downloaded_images = []

    for asset in tqdm(assets, desc="Downloading images"):
        image = download_image(api_key, asset.content)
        format = str(image.format or "")
        filepath = ""
        if output_folder:
            filepath = os.path.join(output_folder, asset.id + "." + format.lower())
            os.makedirs(output_folder, exist_ok=True)
            with open(filepath, "wb") as fp:
                image.save(fp, format)  # type: ignore
        downloaded_images.append(
            DownloadedImage(
                id=asset.id,
                externalId=asset.externalId,
                filepath=filepath or "",
            )
        )
    return downloaded_images


def download_project_text(
    api_key: str,
    assets: List[AssetT],
) -> List[DownloadedText]:
    kili_print("Downloading project text...")
    downloaded_text = []
    for asset in tqdm(assets, desc="Downloading text content"):
        content = download_asset_unicode(api_key, asset.content)
        downloaded_text.append(
            DownloadedText(
                id=asset.id,
                externalId=asset.externalId,
                content=content,
            )
        )
    return downloaded_text
