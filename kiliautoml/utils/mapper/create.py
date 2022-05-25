import os
from typing import Any, List, Optional

import datasets
import kmapper as km
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from img2vec_pytorch import Img2Vec  # type: ignore
from more_itertools import chunked
from PIL.Image import Image as PILImage
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer  # type: ignore

from kiliautoml.utils.constants import InputTypeT
from kiliautoml.utils.download_assets import (
    download_project_images,
    download_project_text,
)
from kiliautoml.utils.helpers import kili_print
from kiliautoml.utils.mapper.clustering import DensityMergeHierarchicalClustering
from kiliautoml.utils.mapper.gudhi_mapper import (
    CoverComplex,
    confusion_filter,
    custom_tooltip_picture,
    custom_tooltip_text,
    data_index_in_mapper,
    gudhi_to_KM,
    topic_score,
)
from kiliautoml.utils.type import AssetStatusT, JobT


def embeddings_text(list_text: List[str]):

    ds = datasets.Dataset.from_pandas(pd.DataFrame({"content": list_text}))  # type: ignore

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment", padding=True
    )
    model = AutoModel.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    # Tokenize sentences
    encoded_input = ds.map(lambda examples: tokenizer(examples["content"]), batched=True)
    encoded_input.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask"]
    )
    encoded_input = torch.utils.data.DataLoader(encoded_input, batch_size=1)  # type: ignore

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    embedings = []
    for input in tqdm(encoded_input):
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, input["attention_mask"])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        embedings.append(sentence_embeddings)

    return np.concatenate((embedings), axis=0)


def embeddings_images(images: List[PILImage], batch_size=4):
    """Get the embeddings of the images using a generic model trained on ImageNet."""
    color_images = [im.convert("RGB") for im in images]
    img2vec = Img2Vec(cuda=torch.cuda.is_available(), model="efficientnet_b7")
    vecs = []
    for imgs in tqdm(list(chunked(color_images, batch_size))):
        _ = np.array(img2vec.get_vec(imgs))
        vecs.append(_)
    return np.concatenate(vecs, axis=0)


class MapperClassification:
    def __init__(
        self,
        *,
        api_key: str,
        input_type: InputTypeT,
        assets: List[Any],
        job: JobT,
        job_name: str,
        assets_repository,  # check type
        asset_status_in: Optional[List[AssetStatusT]],
        focus_class: Optional[List[str]],
    ):
        self.job = job
        self.job_name = job_name
        self.asset_status_in = asset_status_in
        self.assets = assets
        self.focus_class = focus_class
        self.input_type = input_type

        class_list = self.job["content"]["categories"]
        self.cat2id = {}
        for i, cat in enumerate(class_list):
            self.cat2id[cat] = i

        if self.input_type == "IMAGE":
            # Check proper way to create folder
            os.makedirs(assets_repository, exist_ok=True)

            # Get list of image
            self.data = download_project_images(api_key, assets, output_folder=assets_repository)
            kili_print(f"Number of image recovered: {len(self.data)}")

        elif self.input_type == "TEXT":
            # Get list of image
            self.data = download_project_text(api_key, assets)
            kili_print(f"Number of text recovered: {len(self.data)}")

        else:
            raise NotImplementedError

    def create_mapper(
        self,
        cv_folds: int,
    ):
        # Compute embeddings
        kili_print("Computing embeddings")
        embeddings = self._get_embeddings()
        kili_print(f"Embeddings successfully computed with shape: {embeddings.shape}")

        self.assignments: List[List[int]]  # type: ignore
        self.lens_names: List[str]  # type: ignore
        self.lens: np.ndarray  # type: ignore

        self._get_assignments_and_lens(embeddings, cv_folds)

        # Keep only assets in focus_class
        if self.focus_class is not None:
            focus_class_id = [self.cat2id[focus_class] for focus_class in self.focus_class]
            idx_to_keep = [
                idx for idx, label in enumerate(self.lens[:, 1]) if label in focus_class_id
            ]
            self.assignments = [self.assignments[idx] for idx in idx_to_keep]
            self.lens = np.array([self.lens[idx] for idx in idx_to_keep])
            self.data = [self.data[idx] for idx in idx_to_keep]

        kili_print("fitting nodes of Mapper")
        Mapper_kili = CoverComplex(
            complex_type="mapper",  # Always mapper
            input_type="point cloud",  # can be 'point cloud' or 'distance matrix'
            cover="precomputed",  # Always for our type of cover
            assignments=self.assignments,
            colors=None,  # Choose how to colour the nodes. Must be numpy array of dimension 2
            mask=0,  # Remove the clusters containing less than mask points
            clustering=DensityMergeHierarchicalClustering(),  # Choose the clustering algorithm
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
        _ = Mapper_kili.fit(embeddings)
        kili_print(f"Mapper complex fitted with: {len(Mapper_kili.node_info)} nodes")

        idx_points_in_Mapper = data_index_in_mapper(Mapper_kili)
        kili_print(
            f"{round(len(np.unique(idx_points_in_Mapper)) / len(self.assets) * 100, 2)} %"
            " of assets appear in Mapper"
        )

        # Cretae custom tooltip (to be put in custom_tooltip_picture function)
        tooltip_s = self._get_custom_tooltip()

        if self.input_type == "TEXT":
            kili_print("Compute document topic score from list of topic")
            list_text = [downloaded_text.content for downloaded_text in self.data]  # type: ignore
            self.lens = np.column_stack((self.lens, topic_score(list_text)))
            topic_list = ["topic_" + str(i) for i in range(10)]
            self.lens_names = self.lens_names + topic_list

        temp = gudhi_to_KM(Mapper_kili)
        mapper = km.KeplerMapper(verbose=2)
        _ = mapper.visualize(
            temp,
            lens=self.lens,
            lens_names=self.lens_names,
            custom_tooltips=tooltip_s,
            color_values=self.lens,
            color_function_name=self.lens_names,
            title="Mapper_" + self.job_name,
            path_html="Mapper_" + self.job_name + ".html",
        )
        return Mapper_kili

    def _get_embeddings(self):
        if self.input_type == "IMAGE":
            pil_images = [image.image for image in self.data]  # type: ignore
            return embeddings_images(pil_images)

        elif self.input_type == "TEXT":
            list_text = [downloaded_text.content for downloaded_text in self.data]  # type: ignore
            return embeddings_text(list_text)

        else:
            raise NotImplementedError

    def _get_assignments_and_lens(self, embeddings, cv_folds: int):

        if (
            (self.asset_status_in is None)
            or ("TODO" in self.asset_status_in)
            or ("ONGOING" in self.asset_status_in)
        ):  # TODO modify when asset type changed
            self._get_assignments_and_lens_without_labels(embeddings)
        else:  # TODO modify when asset type changed
            self._get_assignments_and_lens_with_labels(embeddings, cv_folds)

    def _get_assignments_and_lens_with_labels(self, embeddings, cv_folds: int):

        # Get labels (as string and number_id)
        labels = [
            asset["labels"][0]["jsonResponse"][self.job_name]["categories"][0]["name"]
            for asset in self.assets
        ]
        label_id_array = [self.cat2id[label] for label in labels]

        # Compute predictions from embeddings with a simple model
        kili_print("Compute prediction by cross-validation with model: linear SVM")
        classifier = make_pipeline(
            StandardScaler(), linear_model.SGDClassifier(loss="log", alpha=0.1)
        )
        predict = cross_val_predict(
            classifier, embeddings, label_id_array, cv=cv_folds, method="predict_proba"
        )
        predict_order = np.argsort(predict, axis=1)
        prediction_true_class = [predict[enum, item] for enum, item in enumerate(label_id_array)]
        predict_class = predict_order[:, -1]
        accuracy = round(np.sum(predict_class == label_id_array) / len(label_id_array) * 100, 2)
        kili_print("Model accuracy is:" f" {accuracy}%")

        # Compute assignments with confusion filter
        self.assignments = confusion_filter(predict, label_id_array)

        # Create lens for statistic displayed in Mapper
        self.lens = np.column_stack(
            (
                prediction_true_class,
                label_id_array,
                np.max(predict, axis=1),
                predict_class,
                predict_class == label_id_array,
            )
        )

        self.lens_names = [
            "confidence_C",
            "correct_class",
            "confidence_PC",
            "predicted_class",
            "prediction_error",
        ]

    def _get_assignments_and_lens_without_labels(self, embeddings):

        idx_labeled_assets = [
            idx for idx, asset in enumerate(self.assets) if len(asset["labels"]) > 0
        ]

        # Get labels (as string and number_id)
        labels = [
            self.assets[idx]["labels"][0]["jsonResponse"][self.job_name]["categories"][0]["name"]
            for idx in idx_labeled_assets
        ]
        label_id_array = [self.cat2id[label] for label in labels]
        # Compute predictions from embeddings with a simple model
        kili_print("Compute prediction with model: linear SVM")
        classifier = make_pipeline(
            StandardScaler(), linear_model.SGDClassifier(loss="log", alpha=0.1)
        )
        classifier.fit(embeddings[idx_labeled_assets, :], label_id_array)
        predict = classifier.predict_proba(embeddings)
        predict_order = np.argsort(predict, axis=1)
        predict_class = predict_order[:, -1]
        accuracy = round(
            np.sum(predict_class[idx_labeled_assets] == label_id_array) / len(label_id_array) * 100,
            2,
        )
        kili_print("Model accuracy (on training data) is:" f" {accuracy}%")

        # Compute assignments with confusion filter
        self.assignments = confusion_filter(predict)

        # Create lens for statistic displayed in Mapper
        self.lens = np.column_stack((np.max(predict, axis=1), predict_class, predict_order[:, -2]))
        self.lens_names = [
            "confidence_PC",
            "predicted_class",
            "alt_predicted_class",
        ]

    def _get_custom_tooltip(self):

        if self.input_type == "IMAGE":
            # with labels available
            if len(self.lens_names) == 5:
                return custom_tooltip_picture(
                    np.column_stack((self.lens[:, 1], self.lens[:, 3])),
                    pict_data_type="img_list",
                    image_list=self.data,
                )
            # without labels available
            else:
                return custom_tooltip_picture(
                    self.lens[:, 1],
                    pict_data_type="img_list",
                    image_list=self.data,
                )

        elif self.input_type == "TEXT":
            # with labels available
            if len(self.lens_names) == 5:
                return custom_tooltip_text(
                    np.column_stack((self.lens[:, 1], self.lens[:, 3])),
                    data=self.data,
                )
            # without labels available
            else:
                return custom_tooltip_text(
                    self.lens[:, 1],
                    data=self.data,
                )
        else:
            raise NotImplementedError
