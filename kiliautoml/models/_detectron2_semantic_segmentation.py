import json
import os
import shutil
from os.path import join
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image
from typing_extensions import Literal, TypedDict

from kiliautoml.models._base_model import BaseModel
from kiliautoml.utils.constants import (
    HOME,
    MLTaskT,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
)
from kiliautoml.utils.download_assets import download_asset_binary
from kiliautoml.utils.helpers import JobPredictions, kili_print
from kiliautoml.utils.path import ModelDirT, Path, PathDetectron2
from kiliautoml.utils.type import AssetT, JobT, LabelMergeStrategyT

setup_logger()


class Detectron2SemanticSegmentationModel(BaseModel):

    ml_task: MLTaskT = "OBJECT_DETECTION"
    model_repository: ModelRepositoryT = "detectron2"

    def __init__(
        self,
        *,
        project_id: str,
        job: JobT,
        job_name: str,
        model_name: ModelNameT,
        model_framework: ModelFrameworkT,
    ):
        # TODO - model_name should be shecked by BaseModel
        if model_name is None:
            model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        BaseModel.__init__(
            self,
            job=job,
            job_name=job_name,
            model_name=model_name,
            model_framework=model_framework,
        )
        self.project_id = project_id

    @staticmethod
    def get_coco_dicts(img_dir):
        """Convert COCO format to Detectron2 format."""
        json_file = os.path.join(img_dir, "labels.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []

        for image in imgs_anns["images"]:
            record = {}

            height, width = image["height"], image["width"]

            record["file_name"] = image["file_name"]
            record["image_id"] = image["id"]
            record["height"] = height
            record["width"] = width

            annotationss = [
                annotations
                for annotations in imgs_anns["annotations"]
                if annotations["image_id"] == image["id"]
            ]
            objs = []
            for annotations in annotationss:
                px = annotations["segmentation"][0][::2]
                py = annotations["segmentation"][0][1::2]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": annotations["category_id"],
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts

    def train(
        self,
        *,
        assets: List[AssetT],
        label_merge_strategy: LabelMergeStrategyT,
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
        disable_wandb: bool,
        verbose: int,
        api_key: str,
    ):
        """Download Kili assets, convert to coco format, then to detectron2 format, train model."""
        _ = verbose
        _ = disable_wandb
        _ = label_merge_strategy

        model_path_repository_dir = Path.model_repository_dir(
            HOME, self.project_id, self.job_name, self.model_repository
        )
        if clear_dataset_cache:
            shutil.rmtree(model_path_repository_dir, ignore_errors=True)
        model_dir = PathDetectron2.append_model_dir(model_path_repository_dir)
        data_dir = PathDetectron2.append_data_dir(model_path_repository_dir)
        eval_dir = PathDetectron2.append_output_evaluation(model_path_repository_dir)

        # 1. Convert to COCO format
        _, classes = convert_kili_semantic_to_coco(
            job_name=self.job_name, assets=assets, output_dir=data_dir, api_key=api_key
        )

        # 2. Convert to Detectron2 format
        MetadataCatalog.clear()
        DatasetCatalog.clear()
        for d in ["train", "val"]:
            # TODO: separate train and test
            DatasetCatalog.register("dataset_" + d, lambda d=d: self.get_coco_dicts(data_dir))
            MetadataCatalog.get("dataset_" + d).set(thing_classes=classes)

        # 3. Train model
        cfg = self.get_cfg_kili(assets, epochs, batch_size, model_dir, classes)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        res = trainer.train()
        print(res)

        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference: # noqa
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, "model_final.pth"
        )  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.00003  # set a custom testing threshold
        predictor = DefaultPredictor(cfg)

        evaluator = COCOEvaluator("dataset_val", output_dir=eval_dir)
        val_loader = build_detection_test_loader(cfg, "dataset_val")  # type:ignore
        eval_res = inference_on_dataset(predictor.model, val_loader, evaluator)
        print(eval_res)
        kili_print(f"Evaluations results are available in {eval_dir}")

        return eval_res["segm"]["AP"]

    def get_cfg_kili(
        self,
        assets: List[AssetT],
        epochs: Optional[int],
        batch_size: int,
        model_dir: ModelDirT,
        classes: List[str],
    ):
        cfg = get_cfg()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.DEVICE = device
        if device == "cpu":
            kili_print("Running on CPU, this will be extremely slow.")
        cfg.merge_from_file(model_zoo.get_config_file(self.model_name))
        cfg.DATASETS.TRAIN = ("dataset_train",)
        cfg.DATASETS.TEST = ("dataset_val",)
        cfg.DATALOADER.NUM_WORKERS = 1
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_name)
        cfg.SOLVER.IMS_PER_BATCH = batch_size  # This is the real "batch size" commonly known to deep learning people # noqa: E501
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        if epochs:
            n_iter = int(epochs * len(assets) / batch_size) + 1
            print("n_iter:", n_iter, "(Recommended min: 500)")
            cfg.SOLVER.MAX_ITER = n_iter  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset # noqa: E501
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512) # noqa: E501
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
            classes
        )  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets) # noqa: E501

        cfg.OUTPUT_DIR = model_dir
        return cfg

    # def visualise_labels(self):
    #     dataset_dicts = self.get_coco_dicts("./output")
    #     dataset_metadata = MetadataCatalog.get("dataset_train")

    #     for d in random.sample(dataset_dicts, 2):
    #         img = cv2.imread(d["file_name"])
    #         visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
    #         out = visualizer.draw_dataset_dict(d)
    #         cv2_imshow(out.get_image()[:, :, ::-1])

    def predict(  # type: ignore
        self,
        *,
        assets: List[AssetT],
        model_path: Optional[str],
        from_project: Optional[str],
        batch_size: int,
        verbose: int,
        clear_dataset_cache: bool,
        api_key: str = "",
    ):
        _ = verbose
        if from_project:
            project_id = from_project
        else:
            project_id = self.project_id
        model_path_repository_dir = Path.model_repository_dir(
            HOME, project_id, self.job_name, self.model_repository
        )
        if clear_dataset_cache:
            shutil.rmtree(model_path_repository_dir, ignore_errors=True)
        model_dir = PathDetectron2.append_model_dir(model_path_repository_dir)
        data_dir = PathDetectron2.append_data_dir(model_path_repository_dir)
        visualization_dir = PathDetectron2.append_output_visualization(model_path_repository_dir)

        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously.
        # We changed it a little bit for inference:
        _, classes = convert_kili_semantic_to_coco(
            job_name=self.job_name, assets=assets, output_dir=data_dir, api_key=api_key
        )
        cfg = self.get_cfg_kili(
            assets, epochs=None, batch_size=batch_size, model_dir=model_dir, classes=classes
        )
        cfg.OUTPUT_DIR = model_dir
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, "model_final.pth"
        )  # path to the model we just trained
        if model_path:
            cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set a custom testing threshold
        predictor = DefaultPredictor(cfg)

        # 2. Convert to Detectron2 format
        MetadataCatalog.clear()
        DatasetCatalog.clear()
        for d in ["train", "val"]:
            # TODO: separate train and test
            DatasetCatalog.register("dataset_" + d, lambda d=d: self.get_coco_dicts(data_dir))
            MetadataCatalog.get("dataset_" + d).set(thing_classes=classes)
        dataset_metadata_train = MetadataCatalog.get("dataset_train")

        # 3. Predict
        dataset_dicts = self.get_coco_dicts(data_dir)
        id_json_list: List[Tuple[str, Dict]] = []  # type: ignore
        for d in dataset_dicts:
            externalId = d["file_name"]  # type:ignore
            print("externalId", externalId)
            im = cv2.imread(externalId)
            outputs = predictor(im)

            annotations = self.get_annotations_from_instances(
                outputs["instances"], class_names=classes
            )
            self.visualize_predictions(visualization_dir, d, dataset_metadata_train, im, outputs)
            id_json_list.append((externalId, {self.job_name: {"annotations": annotations}}))

        job_predictions = JobPredictions(
            job_name=self.job_name,
            external_id_array=[asset["externalId"] for asset in assets],
            json_response_array=[a[1] for a in id_json_list],
            model_name_array=["Kili AutoML"] * len(id_json_list),
            predictions_probability=[100] * len(id_json_list),
        )
        return job_predictions

    @staticmethod
    def get_contours_instance(instances, i: int):
        pred_masks = instances.pred_masks
        masks = np.int8(pred_masks.cpu().detach().numpy())

        im = masks[i] * 255
        im = np.array(im, dtype=np.uint8)
        idx = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        list_x_y = idx[0][0]
        return list_x_y.reshape(-1, 2)

    def get_annotations_from_instances(self, instances, class_names: List[str]):
        """instances contains multiples bbox and object corresponding to one image"""
        annotations = []
        h, w = instances._image_size
        scores = instances.scores.cpu().detach().numpy()
        classes = instances.pred_classes.cpu().detach().numpy()
        for i in range(len(classes)):
            score = scores[i]
            classe = classes[i]
            categories = [Category(name=class_names[classe], confidence=int(score * 100))]
            list_x_y = self.get_contours_instance(instances, i)
            boundingPoly = [
                NormalizedVertices(
                    normalizedVertices=[
                        NormalizedVertice(x=float(round(x / w, 4)), y=float(round(y / h, 4)))
                        for x, y in list_x_y
                    ]
                )
            ]
            annotation = Annotation(
                boundingPoly=boundingPoly,
                mid=None,  # type:ignore
                type="semantic",
                categories=categories,
            )
            annotations.append(annotation)
        return annotations

    def visualize_predictions(self, visualization_dir, d, dataset_metadata_train, im, outputs):
        v = Visualizer(
            im[:, :, ::-1],
            metadata=dataset_metadata_train,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. This option is only available for segmentation models # noqa
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        image_with_predictions = out.get_image()[:, :, ::-1]

        self.save_predictions(visualization_dir, d["file_name"], image_with_predictions)

    def save_predictions(self, visualization_dir, file_name, image_with_predictions):
        im = Image.fromarray(image_with_predictions)
        path = os.path.join(visualization_dir, file_name)
        im.save(path)
        print("predictions image have been saved in", path)

    def find_errors(
        self,
        *,
        assets: List[AssetT],
        label_merge_strategy: LabelMergeStrategyT,
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: int = 0,
        clear_dataset_cache: bool = False,
        api_key: str = "",
    ) -> Any:
        pass


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
    name: str
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


# ## KILI Polygon Semantic Format


class Category(TypedDict):
    name: str
    confidence: int  # between 0 and 100


class NormalizedVertice(TypedDict):
    x: float
    y: float


class NormalizedVertices(TypedDict):
    normalizedVertices: List[NormalizedVertice]


class Annotation(TypedDict):
    boundingPoly: List[NormalizedVertices]
    mid: str
    type: Literal["semantic"]
    categories: List[Category]

    # def check(self):
    #   assert len(self.boundingPoly) == 1


class SemanticJob(TypedDict):
    annotations: List[Annotation]


infos_coco = {
    "year": "2022",
    "version": "1.0",
    "description": "Exported from KiliAutoML",
    "contributor": "KiliAutoML",
    "url": "https://kili-technology.com",
    "date_created": "2022-01-19T09:48:27",
}


def convert_kili_semantic_to_coco(
    job_name: str, assets: List[AssetT], output_dir, api_key: str
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
    category_names = []
    for asset in assets:
        annotations: List[Annotation] = asset["labels"][0]["jsonResponse"][job_name]["annotations"]
        for annotation in annotations:
            categories = annotation["categories"]
            category_names.append(categories[0]["name"])
            categories = annotation["categories"]

    category_name_to_id = {cat_name: i for i, cat_name in enumerate(list(set(category_names)))}
    for cat_name, cat_id in category_name_to_id.items():
        categories_coco: CategoryCoco = {
            "id": cat_id,
            "name": cat_name,
            "supercategory": "",
        }
        labels_json["categories"].append(categories_coco)

    # Fill labels_json
    j = -1
    for i, asset in enumerate(assets):
        annotations_: List[Annotation] = asset["labels"][0]["jsonResponse"][job_name]["annotations"]

        # Add a new image
        img_data = download_asset_binary(api_key, asset["content"])  # jpg
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        file_name = os.path.join(data_dir, f"{i}.jpg")
        cv2.imwrite(file_name, img)
        height = img.shape[0]
        width = img.shape[1]
        image_coco = ImageCoco(
            id=i,
            license=0,
            file_name=file_name,
            height=height,
            width=width,
            date_captured=None,
        )
        labels_json["images"].append(image_coco)

        for annotation in annotations_:
            j += 1
            boundingPoly = annotation["boundingPoly"]
            px: List[float] = [v["x"] * width for v in boundingPoly[0]["normalizedVertices"]]
            py: List[float] = [v["y"] * height for v in boundingPoly[0]["normalizedVertices"]]
            poly_ = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly_ for p in x]

            categories = annotation["categories"]
            cat_id = category_name_to_id[categories[0]["name"]]
            annotations_coco = AnnotationsCoco(
                id=j,
                image_id=i,
                category_id=cat_id,
                bbox=[np.min(px), np.min(py), np.max(px), np.max(py)],
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
    classes: List[str] = list(category_name_to_id.keys())
    return labels_json, classes
