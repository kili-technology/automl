import json
import os
import shutil
import time
from os.path import join
from typing import Dict, List, Tuple

import cv2
import numpy as np
from typing_extensions import TypedDict

from kiliautoml.utils.download_assets import download_asset_binary
from kiliautoml.utils.helpers import get_mapping_category_name_cat_kili_id, kili_print
from kiliautoml.utils.type import AssetT, CategoryIdT, JobNameT, JobT

# ## DETECTRON FORMAT


class ImageCoco(TypedDict):
    id: int
    license: int
    file_name: str
    height: int
    width: int
    date_captured: None


class CategoryCoco(TypedDict):
    id: int
    name: CategoryIdT
    supercategory: str


class AnnotationsCoco(TypedDict):
    id: int
    image_id: int
    category_id: int
    bbox: List[int]
    segmentation: List[List[float]]  # [[x, y, x, y, x ...]]
    area: int
    iscrowd: int


class CocoFormat(TypedDict):
    info: Dict  # type: ignore
    licenses: List[Dict]  # type: ignore
    categories: List[CategoryCoco]
    images: List[ImageCoco]
    annotations: List[AnnotationsCoco]


def convert_kili_semantic_to_coco(
    job_name: JobNameT, assets: List[AssetT], output_dir, api_key: str, job: JobT
) -> Tuple[CocoFormat, List[str]]:
    """
    creates the following structure on the disk:
    <dataset_dir>/
        data/
            <filename0>.<ext>
            <filename1>.<ext>
            ...
        labels.json


    We iterate on the assets and create a coco format for each asset.
    """
    infos_coco = {
        "year": time.strftime("%Y"),
        "version": "1.0",
        "description": "Exported from KiliAutoML",
        "contributor": "KiliAutoML",
        "url": "https://kili-technology.com",
        "date_created": time.strftime("%Y %m %d %H %M"),
    }
    labels_json = CocoFormat(
        info=infos_coco,
        licenses=[],
        categories=[],
        images=[],
        annotations=[],
    )

    # Prepare output folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Mapping category - category id
    mapping_cat_name_cat_kili_id = get_mapping_category_name_cat_kili_id(job)
    cat_kili_ids = list(mapping_cat_name_cat_kili_id.values())
    cat_kili_id_to_coco_id = {categoryId: i for i, categoryId in enumerate(cat_kili_ids)}
    for cat_kili_id, cat_coco_id in cat_kili_id_to_coco_id.items():
        categories_coco: CategoryCoco = {
            "id": cat_coco_id,
            "name": cat_kili_id,
            "supercategory": "",
        }
        labels_json["categories"].append(categories_coco)

    # Fill labels_json
    annotation_j = -1
    for asset_i, asset in enumerate(assets):
        annotations_ = asset.get_annotations_semantic(job_name)["annotations"]

        # Add a new image
        img_data = download_asset_binary(api_key, asset.content)  # jpg
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        file_name = os.path.join(data_dir, f"{asset_i}.jpg")
        cv2.imwrite(file_name, img)
        height = img.shape[0]
        width = img.shape[1]
        image_coco = ImageCoco(
            id=asset_i,
            license=0,
            file_name=file_name,
            height=height,
            width=width,
            date_captured=None,
        )
        labels_json["images"].append(image_coco)

        for annotation in annotations_:
            annotation_j += 1
            boundingPoly = annotation["boundingPoly"]
            px: List[float] = [float(v["x"]) * width for v in boundingPoly[0]["normalizedVertices"]]
            py: List[float] = [
                float(v["y"]) * height for v in boundingPoly[0]["normalizedVertices"]
            ]
            poly_ = [(float(x), float(y)) for x, y in zip(px, py)]
            poly = [p for x in poly_ for p in x]

            categories = annotation["categories"]
            cat_coco_id = cat_kili_id_to_coco_id[categories[0]["name"]]
            annotations_coco = AnnotationsCoco(
                id=annotation_j,
                image_id=asset_i,
                category_id=cat_coco_id,
                bbox=[int(np.min(px)), int(np.min(py)), int(np.max(px)), int(np.max(py))],
                # Objects have only one connected part.
                # But a type of object can appear several times on the same image.
                # The limitation of the single connected part comes from Kili.
                segmentation=[poly],
                area=height * width,
                iscrowd=0,
            )
            labels_json["annotations"].append(annotations_coco)

    with open(join(output_dir, "labels.json"), "w") as outfile:
        json.dump(labels_json, outfile)

    kili_print(f"Kili format has been converted to Coco format. Saved in {output_dir}")
    classes: List[str] = list(cat_kili_id_to_coco_id.keys())
    kili_print("List of classes:", classes)
    return labels_json, classes
