import itertools
import mimetypes
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List

import requests
from kili.queries.asset.helpers import get_file_extension_from_headers
from loguru import logger
from PIL import Image
from PIL.Image import Image as PILImage
from ratelimit import limits, sleep_and_retry
from requests import Response
from tqdm.autonotebook import tqdm

from kiliautoml.utils.helper_mock import GENERATE_MOCK, save_mock_data
from kiliautoml.utils.logging import one_time_logger
from kiliautoml.utils.memoization import kili_memoizer
from kiliautoml.utils.type import AssetExternalIdT, AssetIdT, AssetsLazyList, AssetT


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


class AssetsDownloader:
    use_header = True

    @sleep_and_retry
    @limits(calls=1, period=DELAY)  # type:ignore
    def _throttled_request(
        self, api_key, asset_content, use_header: bool = True, k=0, error=None
    ) -> Response:
        if k == 20:
            raise Exception("Too many retries", error)
        header = {"Authorization": f"X-API-Key: {api_key}"} if use_header else None
        response = requests.get(asset_content, headers=header)
        try:
            assert response.status_code == 200, f"response.status_code: {response.status_code}"

            self.use_header = use_header

            if GENERATE_MOCK:
                id = asset_content.split("/")[-1].split(".")[0]
                save_mock_data(id, response, function_name="throttled_request")
            return response
        except AssertionError as e:
            # Sometimes, the header breaks google bucket and just removing the header makes it work.
            print(e, response, response.status_code, asset_content)
            return self._throttled_request(
                api_key,
                asset_content,
                use_header=not use_header,
                k=k + 1,
                error=[e, response, response.status_code],
            )

    def throttled_request(self, api_key, asset_content):
        """This function enables to call _throttled_request directly with self.use_header

        This can double the speed of downloding sometimes.
        """
        return self._throttled_request(api_key, asset_content, use_header=self.use_header)


asset_downloader = AssetsDownloader()


@kili_memoizer  # We memorize each argument but asset_content wich contains (asset_id + token)
def _throttled_request_memoized(api_key, asset_content, asset_id):
    """Even if asset content varies his token, we use the memoized asset"""
    _ = asset_id
    return asset_downloader.throttled_request(api_key, asset_content)


def throttled_request(api_key, asset_content):
    """
    asset_content contains the id and the token.
    'https://cloud.kili-technology.com/api/label/v2/files?id=3a5aa0ca-328e-4f0f-bedd-4ffff27f796d&token=890cc70e2'

    But we want to memoize the asset even if the token changes.
    """
    if "files?id=" in asset_content and "&token=" in asset_content:
        asset_id = asset_content.split("?id=")[1].split("&token=")[0]
        return _throttled_request_memoized(api_key, asset_content, asset_id)
    elif "?AWSAccessKeyId" in asset_content:
        asset_id = asset_content.split("?AWSAccessKeyId")[0]
        return _throttled_request_memoized(api_key, asset_content, asset_id)
    else:
        one_time_logger("Downloading public asset")  # No security token
        return asset_downloader.throttled_request(api_key, asset_content)


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
    assets: AssetsLazyList,
    output_folder: str,
) -> List[DownloadedImage]:
    logger.info(f"Downloading images to folder {output_folder}")
    downloaded_images = []

    for asset in tqdm(assets, desc="Downloading images"):
        downloaded_images.append(download_and_save_image(api_key, asset, Path(output_folder)))
    return downloaded_images


def download_and_save_image(api_key: str, asset: AssetT, output_folder: Path) -> DownloadedImage:
    extension = get_file_extension_from_headers(asset.content)
    assert extension
    filepath = output_folder / (asset.id + "." + extension)
    if not filepath.exists():
        image = download_image(api_key, asset.content)
        assert image.format
        output_folder.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as f_p:
            image.save(f_p, image.format)

    return DownloadedImage(
        id=asset.id,
        externalId=asset.externalId,
        filepath=str(filepath),
    )


def download_project_text(
    api_key: str,
    assets: AssetsLazyList,
) -> List[DownloadedText]:
    logger.info("Downloading project text...")
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


def get_images_from_local_dataset(
    local_dataset_dir: Path, assets: List[AssetT]
) -> List[DownloadedImage]:
    images = []
    logger.info(f"Loading images from local folder {local_dataset_dir}")
    for asset in assets:
        external_id = asset.externalId
        asset_id = asset.id
        candidate_file_names = itertools.product(
            [external_id, asset_id], _get_extensions_for_type("image")
        )
        for candidate_file_name in candidate_file_names:
            file_path = local_dataset_dir / (candidate_file_name[0] + candidate_file_name[1])
            if file_path.is_file():
                images.append(
                    DownloadedImage(id=asset.id, externalId=external_id, filepath=str(file_path))
                )
                break
    if len(images) == 0:
        raise ValueError(
            f"No files match the external ids of the assets in the directory {local_dataset_dir}"
        )

    return images


def _get_extensions_for_type(general_type):
    for ext in mimetypes.types_map:
        if mimetypes.types_map[ext].split("/")[0] == general_type:
            yield ext
    yield ""
