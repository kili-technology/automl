import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image
from PIL.Image import Image as PILImage
from tqdm import tqdm

from kiliautoml.utils.helpers import kili_print
from kiliautoml.utils.memoization import kili_memoizer


@dataclass
class DownloadedImages:
    id: str
    externalId: str
    filename: str
    image: PILImage


@kili_memoizer
def download_asset_binary(api_key, asset_content):
    asset_data = requests.get(
        asset_content,
        headers={
            "Authorization": f"X-API-Key: {api_key}",
        },
    ).content

    return asset_data


@kili_memoizer
def download_asset_unicode(api_key, asset_content):
    response = requests.get(
        asset_content,
        headers={
            "Authorization": f"X-API-Key: {api_key}",
        },
    )
    assert response.status_code == 200
    text = response.text
    return text


def download_image(api_key, asset_content):
    img_data = download_asset_binary(api_key, asset_content)

    image = Image.open(BytesIO(img_data))
    return image


def download_image_retry(api_key, asset, n_try):
    while n_try < 20:
        try:
            img_data = download_asset_binary(api_key, asset["content"])
            break
        except Exception:
            time.sleep(1)
            n_try += 1
    return img_data  # type:ignore


def download_project_images(
    api_key,
    assets,
    output_folder: Optional[str] = None,
) -> List[DownloadedImages]:
    kili_print("Downloading project images...")
    downloaded_images = []
    for asset in tqdm(assets):
        image = download_image(api_key, asset["content"])
        format = str(image.format or "")

        filename = ""
        if output_folder:
            filename = os.path.join(output_folder, asset["id"] + "." + format.lower())

            with open(filename, "w") as fp:
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
