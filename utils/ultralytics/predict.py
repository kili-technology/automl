import os
from typing import Union, List, Dict, Tuple
import shutil
from glob import glob
import yaml

from utils.helpers import download_project_images, kili_print, build_inference_path
from utils.constants import HOME, ModelFramework, ModelRepository


def ultralytics_predict_object_detection(
    api_key: str,
    assets: Union[List[Dict], List[str]],
    project_id: str,
    model_framework: ModelFramework,
    model_path: str,
    job_name: str,
    verbose: int = 0,
) -> List[Tuple[str, Dict]]:

    if model_framework == ModelFramework.PyTorch:
        filename_weights = "best.pt"
    else:
        raise NotImplementedError(
            f"Predictions with model framework {model_framework} not implemented"
        )

    kili_print(f"Loading model {model_path}")
    kili_print(f"for job {job_name}")
    with open(os.path.join(model_path, "..", "..", "kili.yaml")) as f:
        kili_data_dict = yaml.load(f, Loader=yaml.FullLoader)

    inference_path = build_inference_path(
        HOME, project_id, job_name, ModelRepository.Ultralytics
    )
    model_weights = os.path.join(model_path, filename_weights)

    # path needs to be cleaned-up to avoid inferring unnecessary items.
    if os.path.exists(inference_path) and os.path.isdir(inference_path):
        shutil.rmtree(inference_path)
    os.makedirs(inference_path)

    downloaded_images = download_project_images(api_key, assets, inference_path)

    kili_print("Starting Ultralytics' YoloV5 inference...")
    cmd = (
        f"python detect.py "
        + f'--weights "{model_weights}" '
        + f"--save-txt --save-conf --nosave --exist-ok "
        + f'--source "{inference_path}" --project "{inference_path}"'
    )
    os.system("cd utils/ultralytics/yolov5 && " + cmd)

    inference_files = glob(os.path.join(inference_path, "exp", "labels", "*.txt"))
    inference_files_by_id = {get_id_from_path(pf): pf for pf in inference_files}

    predictions = []
    for image in downloaded_images:
        if image.id in inference_files_by_id:
            kili_predictions = yolov5_to_kili_json(
                inference_files_by_id[image.id], kili_data_dict["names"]
            )
            if verbose >= 1:
                print(f"Asset {image.externalId}: {kili_predictions}")
            predictions.append(
                (image.externalId, {job_name: {"annotations": kili_predictions}})
            )
    return predictions


def get_id_from_path(path_yolov5_inference: str) -> str:
    return os.path.split(path_yolov5_inference)[-1].split(".")[0]


def yolov5_to_kili_json(
    path_yolov5_inference: str, ind_to_categories: List[str]
) -> Dict:

    annotations = []
    with open(path_yolov5_inference, "r") as f:
        for l in f.readlines():
            c, x, y, w, h, p = l.split(" ")
            x, y, w, h = float(x), float(y), float(w), float(h)
            c = int(c)
            p = int(100.0 * float(p))

            annotations.append(
                {
                    "boundingPoly": [
                        {
                            "normalizedVertices": [
                                {"x": x - w / 2, "y": y + h / 2},
                                {"x": x - w / 2, "y": y - h / 2},
                                {"x": x + w / 2, "y": y - h / 2},
                                {"x": x + w / 2, "y": y + h / 2},
                            ]
                        }
                    ],
                    "categories": [{"name": ind_to_categories[c], "confidence": p}],
                    "type": "rectangle",
                }
            )

    return annotations
