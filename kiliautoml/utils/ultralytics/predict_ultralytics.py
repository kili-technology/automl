import os
import shutil
from glob import glob
from typing import Any, Dict, List, Tuple, Union

import yaml
from typing_extensions import TypedDict

from kiliautoml.utils.constants import HOME, ModelFramework, ModelFrameworkT
from kiliautoml.utils.download_assets import download_project_images
from kiliautoml.utils.helpers import JobPredictions, kili_print
from kiliautoml.utils.path import PathUltralytics
from kiliautoml.utils.ultralytics.constants import YOLOV5_REL_PATH


def ultralytics_predict_object_detection(
    api_key: str,
    assets: Union[List[Dict], List[str]],
    project_id: str,
    model_framework: ModelFrameworkT,
    model_path: str,
    job_name: str,
    verbose: int = 0,
    prioritization: bool = False,
) -> JobPredictions:

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

    inference_path = PathUltralytics.inference_dir(HOME, project_id, job_name, "ultralytics")
    model_weights = os.path.join(model_path, filename_weights)

    # path needs to be cleaned-up to avoid inferring unnecessary items.
    if os.path.exists(inference_path) and os.path.isdir(inference_path):
        shutil.rmtree(inference_path)
    os.makedirs(inference_path)

    downloaded_images = download_project_images(
        api_key, assets, project_id, output_folder=inference_path
    )
    # https://github.com/ultralytics/yolov5/blob/master/detect.py
    # default  --conf-thres=0.25, --iou-thres=0.45
    prioritizer_args = " --conf-thres=0.01  --iou-thres=0.45 " if prioritization else ""

    kili_print("Starting Ultralytics' YoloV5 inference...")
    cmd = (
        "python detect.py "
        + f'--weights "{model_weights}" '
        + "--save-txt --save-conf --nosave --exist-ok "
        + f'--source "{inference_path}" --project "{inference_path}" '
        + prioritizer_args
    )
    os.system("cd " + YOLOV5_REL_PATH + " && " + cmd)

    inference_files = glob(os.path.join(inference_path, "exp", "labels", "*.txt"))
    inference_files_by_id = {get_id_from_path(pf): pf for pf in inference_files}

    kili_print("Converting Ultralytics' YoloV5 inference to Kili JSON format...")
    id_json_list: List[Tuple[str, Dict]] = []

    proba_list: List[float] = []
    for image in downloaded_images:
        if image.id in inference_files_by_id:
            kili_predictions, probabilities = yolov5_to_kili_json(
                inference_files_by_id[image.id], kili_data_dict["names"]
            )
            proba_list.append(min(probabilities))
            if verbose >= 1:
                print(f"Asset {image.externalId}: {kili_predictions}")
            id_json_list.append((image.externalId, {job_name: {"annotations": kili_predictions}}))

    # TODO: move this check in the prioritizer
    if len(id_json_list) < len(downloaded_images):
        kili_print(
            "WARNING: Not enouth predictions. Missing prediction for"
            f" {len(downloaded_images) - len(id_json_list)} assets."
        )
        if prioritization:
            # TODO: Automatically tune the threshold
            # TODO: Do not crash and put 0 probability for missing assets.
            raise Exception(
                "Not enough predictions for prioritization. You should either train longer the"
                " model, or lower the --conf-thres "
            )

    job_predictions = JobPredictions(
        job_name=job_name,
        external_id_array=[a[0] for a in id_json_list],
        json_response_array=[a[1] for a in id_json_list],
        model_name_array=["Kili AutoML"] * len(id_json_list),
        predictions_probability=proba_list,
    )
    print("predictions_probability", job_predictions.predictions_probability)
    return job_predictions


def get_id_from_path(path_yolov5_inference: str) -> str:
    return os.path.split(path_yolov5_inference)[-1].split(".")[0]


class CategoryNameConfidence(TypedDict):
    name: str
    # confidence is a probability between 0 and 100.
    confidence: int


class BBoxAnnotation(TypedDict):
    boundingPoly: Any
    categories: List[CategoryNameConfidence]
    type: str


def yolov5_to_kili_json(
    path_yolov5_inference: str, ind_to_categories: List[str]
) -> Tuple[List[BBoxAnnotation], List[int]]:
    """Returns a list of annotations and of probabilities"""
    annotations = []
    probabilities = []
    with open(path_yolov5_inference, "r") as f:
        for line in f.readlines():
            c_, x_, y_, w_, h_, p_ = line.split(" ")
            x, y, w, h = float(x_), float(y_), float(w_), float(h_)
            c = int(c_)
            p = int(100.0 * float(p_))

            category: CategoryNameConfidence = {
                "name": ind_to_categories[c],
                "confidence": p,
            }
            probabilities.append(p)

            bbox_annotation: BBoxAnnotation = {
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
                "categories": [category],
                "type": "rectangle",
            }

            annotations.append(bbox_annotation)

    return (annotations, probabilities)
