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
from tqdm import tqdm

from kiliautoml.utils.helper_mock import GENERATE_MOCK, save_mock_data
from kiliautoml.utils.helpers import kili_print
from kiliautoml.utils.memoization import kili_memoizer


@dataclass
class DownloadedImages:
    id: str
    externalId: str
    filename: str
    image: PILImage


@dataclass
class DownloadedText:
    id: str
    externalId: str
    content: str


DELAY = 60 / 250  # 250 calls per minutes


@sleep_and_retry
@limits(calls=1, period=DELAY)
def throttled_request(api_key, asset_content, use_header=True, k=0) -> Response:  # type: ignore
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
        print(e)
        return throttled_request(api_key, asset_content, use_header=not use_header, k=k + 1)


@kili_memoizer
def download_asset_binary(api_key, asset_content):
    response = throttled_request(api_key, asset_content)
    asset_data = response.content
    return asset_data


@kili_memoizer
def download_asset_unicode(api_key, asset_content):
    response = throttled_request(api_key, asset_content)
    text = response.text
    return text


def download_image(api_key, asset_content):
    img_data = download_asset_binary(api_key, asset_content)

    image = Image.open(BytesIO(img_data))
    return image


def download_image_retry(api_key, asset, n_try: int):
    while n_try < 20:
        try:
            img_data = download_asset_binary(api_key, asset["content"])
            break
        except Exception:
            time.sleep(1)
            n_try += 1
    return img_data  # type:ignore


def download_project_images(
    api_key: str,
    assets,
    output_folder: Optional[str] = None,
) -> List[DownloadedImages]:
    kili_print("Downloading images to folder {}".format(output_folder))
    downloaded_images = []

    for asset in tqdm(assets, desc="Downloading images"):
        image = download_image(api_key, asset["content"])
        format = str(image.format or "")
        filename = ""
        if output_folder:
            filename = os.path.join(output_folder, asset["id"] + "." + format.lower())
            os.makedirs(output_folder, exist_ok=True)
            with open(filename, "wb") as fp:
                image.save(fp, format)  # type: ignore
        downloaded_images.append(
            DownloadedImages(
                id=asset["id"],
                externalId=asset["externalId"],
                filename=filename or "",
                image=image,
            )
        )
    return downloaded_images


def download_project_text(
    api_key: str,
    assets,
) -> List[DownloadedText]:
    kili_print("Downloading project text...")
    downloaded_text = []
    for asset in tqdm(assets, desc="Downloading text content"):
        content = download_asset_unicode(api_key, asset["content"])
        downloaded_text.append(
            DownloadedText(
                id=asset["id"],
                externalId=asset["externalId"],
                content=content,
            )
        )
    return downloaded_text
