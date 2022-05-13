import os
from typing import Any, List  # , Optional

import kmapper as km
import numpy as np
import torch
from img2vec_pytorch import Img2Vec
from more_itertools import chunked
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from kiliautoml.utils.download_assets import DownloadedImages, download_project_images
from kiliautoml.utils.helpers import kili_print
from kiliautoml.utils.mapper.clustering import DensityMergeHierarchicalClustering
from kiliautoml.utils.mapper.gudhi_mapper import (
    CoverComplex,
    confusion_filter,
    custom_tooltip_picture,
    data_index_in_mapper,
    gudhi_to_KM,
)
from kiliautoml.utils.type import LabelTypeT


def embeddings_downloaded_images(images: List[DownloadedImages], batch_size=4) -> np.ndarray:
    """Get the embeddings of the images using a generic model trained on ImageNet."""
    color_images = [im.image.convert("RGB") for im in images]
    img2vec = Img2Vec(cuda=torch.cuda.is_available(), model="efficientnet_b7")
    vecs = []
    for imgs in tqdm(list(chunked(color_images, batch_size))):
        _ = np.array(img2vec.get_vec(imgs))
        vecs.append(_)
    return np.concatenate(vecs, axis=0)


class MapperImageClassification:
    def __init__(
        self,
        *,
        api_key: str,
        project_id: str,
        assets: List[Any],
        job: dict,
        job_name: str,
        assets_repository,  # check type
        label_types: LabelTypeT,
    ):
        self.job = job
        self.job_name = job_name
        self.label_types = label_types
        self.assets = assets

        class_list = self.job["content"]["categories"]
        self.cat2id = {}
        for i, cat in enumerate(class_list):
            self.cat2id[cat] = i

        # Check proper way to create folder
        os.makedirs(assets_repository, exist_ok=True)

        # Get list of image
        self.image_list = download_project_images(
            api_key, assets, project_id, output_folder=assets_repository
        )
        kili_print(f"Number of image recovered: {len(self.image_list)}")

    def create_mapper(
        self,
        cv_folds: int,
    ):

        # Compute embeddings
        kili_print("Computing embeddings")
        self.embeddings = embeddings_downloaded_images(self.image_list)
        kili_print(f"Embeddings successfully computed with shape: {self.embeddings.shape}")

        if self.label_types is None:  # TODO modify when asset type changed
            assignments, lens, lens_names = self._get_assignments_and_lens_with_labels(cv_folds)
        else:  # TODO modify when asset type changed
            assignments, lens, lens_names = self._get_assignments_and_lens_without_labels(cv_folds)

        kili_print("fitting nodes of Mapper")
        Mapper_kili = CoverComplex(
            complex_type="mapper",  # Always Mapper
            # can be 'point cloud' or 'distance matrix' if already computed
            input_type="point cloud",
            cover="precomputed",  # Always for our type of cover
            assignments=assignments,
            colors=None,  # Choose how to colour the nodes. Must be numpy array of dimension 2
            mask=0,  # Remove the clusters containing less than mask points
            # Choose the clustering algorithm
            clustering=DensityMergeHierarchicalClustering(),
            filters=None,  # Choose the filter. Must be numpy array of dimension 2
            filter_bnds=None,  # Filter bounds, does not have to be specified
            resolutions=None,  # number of elements in which we cut the filter's output
            gains=None,  # proportion of overlap between filter's bounds
            input_name="Mapper",
            cover_name="confusion",
            color_name="None",  # Names for the outputs
            verbose=True,
        )

        # Fit the cover complex on input data
        _ = Mapper_kili.fit(self.embeddings)
        kili_print(f"Mapper complex fitted with: {len(Mapper_kili.node_info)} nodes")

        idx_points_in_Mapper = data_index_in_mapper(Mapper_kili)
        kili_print(
            f"{round(len(np.unique(idx_points_in_Mapper)) / len(self.label_id_array) * 100, 2)} %"
            " of assets appear in Mapper"
        )

        temp = gudhi_to_KM(Mapper_kili)
        mapper = km.KeplerMapper(verbose=2)
        _ = mapper.visualize(
            temp,
            lens=lens,
            lens_names=lens_names,
            custom_tooltips=self.tooltip_s,
            color_values=lens,
            color_function_name=lens_names,
            title="Mapper_" + self.job_name,
            path_html="Mapper_" + self.job_name + ".html",
        )
        return Mapper_kili

    def _get_assignments_and_lens_with_labels(self, cv_folds: int):

        # Get labels (as string and number_id)
        self.labels = [
            asset["labels"][0]["jsonResponse"][self.job_name]["categories"][0]["name"]
            for asset in self.assets
        ]
        self.label_id_array = [self.cat2id[label] for label in self.labels]

        # Compute predictions from embeddings with a simple model
        kili_print("Compute prediction with model: linear SVM")
        classifier = make_pipeline(
            StandardScaler(), linear_model.SGDClassifier(loss="log", alpha=0.1)
        )
        self.predict = cross_val_predict(
            classifier, self.embeddings, self.label_id_array, cv=cv_folds, method="predict_proba"
        )
        self.predict_order = np.argsort(self.predict, axis=1)
        self.prediction_true_class = [
            self.predict[enum, item] for enum, item in enumerate(self.label_id_array)
        ]
        self.predict_class = self.predict_order[:, -1]
        accuracy = round(
            np.sum(self.predict_class == self.label_id_array) / len(self.label_id_array) * 100, 2
        )
        kili_print("Model accuracy is:" f" {accuracy}%")

        # Cretae custom tooltip (to be put in custom_tooltip_picture function)
        self.tooltip_s = custom_tooltip_picture(
            np.column_stack((self.label_id_array, self.predict_class)),
            pict_data_type="img_list",
            image_list=self.image_list,
        )

        # Compute assignments with confusion filter
        assignments = confusion_filter(self.predict, self.label_id_array)

        # Create lens for statistic displayed in Mapper
        lens = np.column_stack(
            (
                self.prediction_true_class,
                self.label_id_array,
                np.max(self.predict, axis=1),
                self.predict_class,
                self.predict_class == self.label_id_array,
            )
        )
        lens_names = [
            "confidence_C",
            "correct_class",
            "confidence_PC",
            "predicted_class",
            "prediction_error",
        ]

        return assignments, lens, lens_names

    def _get_assignments_and_lens_without_labels(self, cv_folds: int):

        idx_labeled_assets = [
            idx for idx, asset in enumerate(self.assets) if len(asset["labels"]) > 0
        ]

        # Get labels (as string and number_id)
        self.labels = [
            self.assets[idx]["labels"][0]["jsonResponse"][self.job_name]["categories"][0]["name"]
            for idx in idx_labeled_assets
        ]
        self.label_id_array = [self.cat2id[label] for label in self.labels]

        # Compute predictions from embeddings with a simple model
        kili_print("Compute prediction with model: linear SVM")
        classifier = make_pipeline(
            StandardScaler(), linear_model.SGDClassifier(loss="log", alpha=0.1)
        )
        classifier.fit(self.embeddings[idx_labeled_assets, :], self.label_id_array)
        self.predict = classifier.predict(self.embeddings)
        self.predict_order = np.argsort(self.predict, axis=1)
        self.prediction_true_class = [
            self.predict[enum, item] for enum, item in enumerate(self.label_id_array)
        ]
        self.predict_class = self.predict_order[:, -1]
        accuracy = round(
            np.sum(self.predict_class == self.label_id_array) / len(self.label_id_array) * 100, 2
        )
        kili_print("Model accuracy is:" f" {accuracy}%")

        # Cretae custom tooltip (to be put in custom_tooltip_picture function)
        self.tooltip_s = custom_tooltip_picture(
            self.predict_class,
            pict_data_type="img_list",
            image_list=self.image_list,
        )

        # Compute assignments with confusion filter
        assignments = confusion_filter(self.predict)

        # Create lens for statistic displayed in Mapper
        lens = np.column_stack(
            (np.max(self.predict, axis=1), self.predict_class, self.predict_order[:, -2])
        )
        lens_names = [
            "confidence_PC",
            "predicted_class",
            "alt_predicted_class",
        ]

        return assignments, lens, lens_names
