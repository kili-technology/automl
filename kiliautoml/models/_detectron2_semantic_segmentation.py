import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

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
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm
from typing_extensions import Literal

from kiliautoml.models._base_model import (
    BaseInitArgs,
    KiliBaseModel,
    ModalTrainArgs,
    ModelConditions,
)
from kiliautoml.utils.detectron2.utils_detectron import (
    CocoFormat,
    convert_kili_semantic_to_coco,
)
from kiliautoml.utils.download_assets import download_project_images
from kiliautoml.utils.helper_label_error import find_all_label_errors
from kiliautoml.utils.helpers import categories_from_job
from kiliautoml.utils.logging import logger
from kiliautoml.utils.path import ModelDirT, Path, PathDetectron2
from kiliautoml.utils.type import (
    AssetsLazyList,
    CategoryIdT,
    CategoryT,
    JobNameT,
    JobPredictions,
    JsonResponseSemanticT,
    KiliSemanticAnnotation,
    NormalizedVertice,
    NormalizedVertices,
    ProjectIdT,
    VerboseLevelT,
)

setup_logger()


class Detectron2SemanticSegmentationModel(KiliBaseModel):
    model_conditions = ModelConditions(
        ml_task="OBJECT_DETECTION",
        model_repository="detectron2",
        possible_ml_backend=["pytorch"],
        advised_model_names=[
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        ],
        input_type="IMAGE",
        content_input="radio",
        tools=["semantic", "polygon"],
    )

    def __init__(
        self,
        *,
        base_init_args: BaseInitArgs,
    ):
        KiliBaseModel.__init__(self, base_init_args)

    @staticmethod
    def _convert_coco_to_detectron(img_dir, mode: Literal["train", "test"]):
        """Convert COCO format to Detectron2 format."""
        json_file = os.path.join(img_dir, "labels.json")
        with open(json_file) as f:
            imgs_anns: CocoFormat = json.load(f)

        dataset_dicts = []

        images_train, images_test = train_test_split(
            imgs_anns["images"], random_state=42, shuffle=False
        )
        list_images = images_train if mode == "train" else images_test
        for image in list_images:
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
        assets: AssetsLazyList,
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
        disable_wandb: bool,
        verbose: VerboseLevelT,
        modal_train_args: ModalTrainArgs,
    ):
        """Download Kili assets, convert to coco format, then to detectron2 format, train model."""
        _ = verbose, modal_train_args
        if not disable_wandb:
            logger.warning(
                "Wandb is not yet available on Detectron2. But tensorboard is available."
            )

        model_path_repository_dir = (
            Path.model_repository_dir(  # TODO: Use instead self.model_repository_dir
                self.project_id, self.job_name, self.model_conditions.model_repository
            )
        )
        if clear_dataset_cache:
            shutil.rmtree(model_path_repository_dir, ignore_errors=True)
        model_dir = PathDetectron2.append_model_dir(model_path_repository_dir)
        data_dir = PathDetectron2.append_data_dir(model_path_repository_dir)
        eval_dir = PathDetectron2.append_output_evaluation(model_path_repository_dir)

        # 1. Convert to COCO format
        full_classes = categories_from_job(job=self.job)
        _, _classes = convert_kili_semantic_to_coco(
            job_name=self.job_name,
            assets=assets,
            output_dir=data_dir,
            api_key=self.api_key,
            job=self.job,
        )

        assert len(set(full_classes)) == len(full_classes)
        if len(_classes) < len(full_classes):
            logger.warning(
                f"Your training set contains only {len(_classes)} classes whereas the"
                f" ontology contains {len(full_classes)} classes."
            )

        # 2. Convert to Detectron2 format
        MetadataCatalog.clear()
        DatasetCatalog.clear()
        for d in ["train", "val"]:
            # TODO: separate train and test
            DatasetCatalog.register(
                "dataset_" + d, lambda d=d: self._convert_coco_to_detectron(data_dir, mode=d)
            )
            MetadataCatalog.get("dataset_" + d).set(thing_classes=full_classes)

        # 3. Train model
        cfg = self._get_cfg_kili(assets, epochs, batch_size, model_dir, full_classes)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)
        train_res = trainer.train()
        logger.info("Training metrics", train_res)

        # 4. Inference
        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference: # noqa
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, "model_final.pth"
        )  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set a custom testing threshold
        predictor = DefaultPredictor(cfg)

        evaluator = COCOEvaluator("dataset_val", output_dir=eval_dir)
        val_loader = build_detection_test_loader(cfg, "dataset_val")  # type:ignore
        eval_res = inference_on_dataset(predictor.model, val_loader, evaluator)
        logger.info(eval_res)
        logger.info(f"Evaluations results are available in {eval_dir}")
        logger.info("The logs and model are saved in ", cfg.OUTPUT_DIR)

        if "segm" not in eval_res:
            logger.warning(
                "No prediction in the training evaluation: your Epoch number may be too low."
            )
        return eval_res

    def _get_cfg_kili(
        self,
        assets: AssetsLazyList,
        epochs: Optional[int],
        batch_size: int,
        model_dir: ModelDirT,
        classes: List[CategoryIdT],
    ):
        cfg = get_cfg()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.DEVICE = device
        if device == "cpu":
            logger.warning("Running on CPU, this will be extremely slow.")
        cfg.merge_from_file(model_zoo.get_config_file(self.model_name))
        cfg.DATASETS.TRAIN = ("dataset_train",)
        cfg.DATASETS.TEST = ("dataset_val",)
        cfg.DATALOADER.NUM_WORKERS = 1
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_name)
        cfg.SOLVER.IMS_PER_BATCH = batch_size  # This is the real "batch size" commonly known to deep learning people # noqa: E501
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.TEST.EVAL_PERIOD = 100
        if epochs:
            n_iter = int(epochs * len(assets) / batch_size) + 1
            logger.info("n_iter:", n_iter, "(Recommended min: 500)")
            cfg.SOLVER.MAX_ITER = n_iter
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
            classes
        )  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets) # noqa: E501

        cfg.OUTPUT_DIR = model_dir
        logger.info("The model and the logs will be be saved in ", cfg.OUTPUT_DIR)
        return cfg

    def predict(
        self,
        *,
        assets: AssetsLazyList,
        model_path: Optional[str],
        from_project: Optional[ProjectIdT],
        batch_size: int,
        verbose: VerboseLevelT,
        clear_dataset_cache: bool,
    ):
        _ = verbose
        if from_project:
            project_id = from_project
        else:
            project_id = self.project_id
        model_path_repository_dir = Path.model_repository_dir(
            project_id, self.job_name, self.model_conditions.model_repository
        )
        if clear_dataset_cache:
            shutil.rmtree(model_path_repository_dir, ignore_errors=True)
        model_dir = PathDetectron2.append_model_dir(model_path_repository_dir)
        data_dir = PathDetectron2.append_data_dir(model_path_repository_dir)
        visualization_dir = PathDetectron2.append_output_visualization(model_path_repository_dir)

        downloaded_images = download_project_images(
            api_key=self.api_key, assets=assets, output_folder=data_dir
        )

        full_classes = categories_from_job(job=self.job)
        assert len(set(full_classes)) == len(full_classes)

        cfg = self._get_cfg_kili(
            assets, epochs=None, batch_size=batch_size, model_dir=model_dir, classes=full_classes
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
        MetadataCatalog.get("dataset_val").set(thing_classes=full_classes)
        dataset_metadata_predict = MetadataCatalog.get("dataset_val")

        # 3. Predict
        id_json_list: List[Tuple[str, Dict[JobNameT, JsonResponseSemanticT]]] = []  # type: ignore

        for image in tqdm(downloaded_images, desc="Predicting"):
            im = cv2.imread(image.filepath)
            outputs = predictor(im)

            annotations = self.get_annotations_from_instances(
                outputs["instances"], class_names=full_classes
            )
            self._visualize_predictions(
                visualization_dir, image.filepath, dataset_metadata_predict, im, outputs
            )
            semantic_job: JsonResponseSemanticT = {"annotations": annotations}
            id_json_list.append((image.externalId, {self.job_name: semantic_job}))

        json_response_array = [a[1] for a in id_json_list]
        job_predictions = JobPredictions(
            job_name=self.job_name,
            external_id_array=[asset.externalId for asset in assets],
            json_response_array=json_response_array,  # type:ignore
            model_name_array=["Kili AutoML"] * len(id_json_list),
            predictions_probability=[100] * len(id_json_list),
        )
        return job_predictions

    @staticmethod
    def get_contours_instance(instances, i: int):
        """Output of detectron is an image. We only want the contours."""
        pred_masks = instances.pred_masks
        masks = np.int8(pred_masks.cpu().detach().numpy())

        im = masks[i] * 255  # type:ignore
        im = np.array(im, dtype=np.uint8)
        idx = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        list_x_y = idx[0][0]
        list_x_y = list_x_y.reshape(-1, 2)

        def purge(x_y):
            """We project point 1 to the [point 1 - point 2] axis"""
            # x_y = np.roll(x_y, 1, axis=0)
            x_y_matrix = np.zeros((3, len(x_y) - 2, 2))
            x_y_matrix[0] = x_y[0:-2]  # point 0
            x_y_matrix[1] = x_y[1:-1]  # point 1
            x_y_matrix[2] = x_y[2:]  # point 2

            vectors_1 = x_y_matrix[1] - x_y_matrix[0]
            vectors_2 = x_y_matrix[2] - x_y_matrix[0]

            inner = np.einsum("ai,ai->a", vectors_1, vectors_2)

            L2_v1 = np.linalg.norm(vectors_1, axis=1)
            L2_v2 = np.linalg.norm(vectors_2, axis=1)
            cos = inner / (L2_v1 * L2_v2)

            new_vectors_1 = vectors_2 * np.stack([cos, cos], axis=1)

            new_point_1 = x_y_matrix[0] + new_vectors_1

            projection_vector = new_point_1 - x_y_matrix[1]

            # If the difference is more than a pixel, keep the intermediate point
            norm = np.linalg.norm(projection_vector, axis=1)

            diameter = np.max(x_y[:, 0]) - np.min(x_y[:, 0]) + np.max(x_y[:, 1]) - np.min(x_y[:, 1])
            keep_points = (
                norm > diameter / 10
            )  # the more the value, the more granular the prediction

            # We do not want to delete more than half the points in a row
            keep_points[::2] = True
            mask = np.array([True] * len(x_y))
            mask[1:-1] = keep_points
            return x_y[mask]

        for _ in range(10):
            list_x_y = purge(list_x_y)
        return list_x_y

    def get_annotations_from_instances(self, instances, class_names: List[CategoryIdT]):
        """instances contains multiples bbox and object corresponding to one image"""
        annotations = []
        h, w = instances._image_size
        scores = instances.scores.cpu().detach().numpy()
        classes = instances.pred_classes.cpu().detach().numpy()
        for class_i in range(len(classes)):
            score = scores[class_i]
            classe = classes[class_i]
            categories = [CategoryT(name=class_names[classe], confidence=int(score * 100))]
            list_x_y = self.get_contours_instance(instances, class_i)

            boundingPoly = [
                NormalizedVertices(
                    normalizedVertices=[
                        NormalizedVertice(x=float(round(x / w, 4)), y=float(round(y / h, 4)))
                        for x, y in list_x_y
                    ]
                )
            ]
            annotation = KiliSemanticAnnotation(
                boundingPoly=boundingPoly,
                mid=None,  # type:ignore
                type="semantic",
                categories=categories,
            )
            annotations.append(annotation)
        return annotations

    def _visualize_predictions(
        self, visualization_dir, file_name, dataset_metadata_train, im, outputs
    ):
        v = Visualizer(
            im[:, :, ::-1],
            metadata=dataset_metadata_train,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW,
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        image_with_predictions = out.get_image()[:, :, ::-1]

        self.save_predictions(visualization_dir, file_name, image_with_predictions)

    def save_predictions(self, visualization_dir, file_name, image_with_predictions):
        im = Image.fromarray(image_with_predictions)
        path = os.path.join(visualization_dir, file_name)
        im.save(path)
        logger.info("predictions image have been saved in", path)

    def find_errors(
        self,
        *,
        assets: AssetsLazyList,
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: VerboseLevelT,
        clear_dataset_cache: bool = False,
    ):
        _ = cv_n_folds
        _ = epochs

        job_predictions = self.predict(
            assets=assets,
            model_path=None,
            from_project=None,
            batch_size=batch_size,
            verbose=verbose,
            clear_dataset_cache=clear_dataset_cache,
        )

        return find_all_label_errors(
            assets=assets,
            json_response_array=job_predictions.json_response_array,
            external_id_array=job_predictions.external_id_array,
            job_name=self.job_name,
            ml_task=self.model_conditions.ml_task,
            tool="semantic",
        )
