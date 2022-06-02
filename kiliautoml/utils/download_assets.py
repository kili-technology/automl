import os
import shutil
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


ONE_MINUTE = 60


@sleep_and_retry
@limits(calls=250, period=ONE_MINUTE)
def throttled_request(api_key, asset_content) -> Response:  # type: ignore
    for _ in range(20):
        try:
            response = requests.get(
                asset_content,
                headers={
                    "Authorization": f"X-API-Key: {api_key}",
                },
            )
            assert response.status_code == 200
            return response
        except Exception:
            time.sleep(1)


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


def download_project_image_clean_lab(*, assets, api_key, data_path, job_name):
    """
    Download assets that are stored in Kili and save them to folders depending on their
    label category
    """
    shutil.rmtree(data_path, ignore_errors=True)

    for asset in tqdm(assets, desc="Downloading images"):
        img_data = download_asset_binary(api_key, asset["content"])
        img_name = asset["labels"][0]["jsonResponse"][job_name]["categories"][0]["name"]
        img_path = os.path.join(data_path, img_name)
        os.makedirs(img_path, exist_ok=True)
        with open(os.path.join(img_path, asset["id"] + ".jpg"), "wb") as handler:
            handler.write(img_data)
