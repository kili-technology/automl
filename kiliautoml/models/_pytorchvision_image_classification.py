import os
import pathlib
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as torch_Data
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.autonotebook import tqdm

from kiliautoml.models._base_model import (
    BaseInitArgs,
    KiliBaseModel,
    ModelConditions,
    ModelTrainArgs,
)
from kiliautoml.utils.download_assets import (
    download_project_images,
    get_images_from_local_dataset,
)
from kiliautoml.utils.helper_label_error import ErrorRecap, LabelingError
from kiliautoml.utils.logging import logger
from kiliautoml.utils.path import Path, PathPytorchVision
from kiliautoml.utils.pytorchvision.image_classification import (
    ClassificationPredictDataset,
    ClassificationTrainDataset,
    data_transforms,
    get_trained_model_image_classif,
    initialize_model_img_class,
    predict_probabilities,
)
from kiliautoml.utils.type import (
    AssetsLazyList,
    JobPredictions,
    JsonResponseClassification,
    ModelNameT,
    ProjectIdT,
)


class PyTorchVisionImageClassificationModel(KiliBaseModel):
    model_conditions = ModelConditions(
        ml_task="CLASSIFICATION",
        model_repository="torchvision",
        possible_ml_backend=["pytorch"],
        advised_model_names=[
            ModelNameT("efficientnet_b0"),
            ModelNameT("resnet50"),
        ],
        input_type="IMAGE",
        content_input="radio",
        tools=None,
    )

    def __init__(
        self,
        *,
        base_init_args: BaseInitArgs,
    ) -> None:
        KiliBaseModel.__init__(self, base_init_args)

        # To set to False if the input size varies a lot and you see that the training takes
        # too much time
        cudnn.benchmark = True

        model_repository_dir = self.model_repository_dir
        self.model_dir = PathPytorchVision.append_model_dir(model_repository_dir)
        self.model_path = PathPytorchVision.append_model_path(model_repository_dir, self.model_name)
        self.data_dir = PathPytorchVision.append_data_dir(model_repository_dir)

        # TODO: The list of classes the model has to deal with should be stored during
        # the initialization of each model, and not just for PyTorchVisionImageClassificationModel
        self.class_name_to_idx = {
            category: i for i, category in enumerate(base_init_args["job"]["content"]["categories"])
        }
        self.class_names = list(self.class_name_to_idx.keys())

    def train(
        self,
        *,
        assets: AssetsLazyList,
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
        disable_wandb: bool,
        api_key: str = "",
        model_train_args: ModelTrainArgs,
        local_dataset_dir: Optional[pathlib.Path],
    ):
        _ = clear_dataset_cache, model_train_args
        if local_dataset_dir is None:
            images = download_project_images(
                api_key=api_key, assets=assets, output_folder=self.data_dir
            )
        else:
            images = get_images_from_local_dataset(local_dataset_dir, assets.assets)
        labels = []
        for asset in assets:
            labels.append(
                asset.get_annotations_classification(self.job_name)["categories"][0]["name"]
            )

        splits = {}
        splits["train"], splits["val"] = train_test_split(
            range(len(labels)), test_size=0.3, random_state=42
        )
        label_train = [labels[i] for i in splits["train"]]
        label_val = [labels[i] for i in splits["val"]]

        if len(np.unique(label_train)) + len(np.unique(label_val)) < 2 * len(self.class_names):
            raise Exception(
                "Some category are not represented in train or val dataset, increase sample size"
            )

        image_datasets = {
            x: ClassificationTrainDataset(
                [images[i] for i in splits[x]],
                [labels[i] for i in splits[x]],
                self.class_name_to_idx,
                data_transforms[x],
            )
            for x in ["train", "val"]
        }

        _, model_evaluation = get_trained_model_image_classif(
            epochs=epochs,
            model_name=self.model_name,  # type: ignore
            batch_size=batch_size,
            category_ids=self.class_names,
            image_datasets=image_datasets,
            save_model_path=self.model_path,
            disable_wandb=disable_wandb,
        )
        return model_evaluation

    def eval(
        self,
        *,
        assets: AssetsLazyList,
        batch_size: int,
        clear_dataset_cache: bool = False,
        model_path: Optional[str],
        from_project: Optional[ProjectIdT],
        local_dataset_dir: Optional[pathlib.Path],
    ):
        raise NotImplementedError("Evaluation is not implemented for Image Classification yet.")

    def predict(
        self,
        *,
        assets: AssetsLazyList,
        model_path: Optional[str],
        from_project: Optional[ProjectIdT],
        batch_size: int,
        clear_dataset_cache: bool,
        api_key: str = "",
        local_dataset_dir: Optional[pathlib.Path],
    ):
        _ = clear_dataset_cache

        if local_dataset_dir is None:
            images = download_project_images(
                api_key=api_key, assets=assets, output_folder=self.data_dir
            )
        else:
            images = get_images_from_local_dataset(local_dataset_dir, assets.assets)

        dataset = ClassificationPredictDataset(images, data_transforms["val"])

        model = initialize_model_img_class(
            self.model_name,  # type: ignore
            self.class_names,
        )

        model_path = self.get_model_path(model_path, from_project)

        model.load_state_dict(torch.load(model_path))
        loader = torch_Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        prob_arrays = predict_probabilities(loader, model)

        job_predictions = JobPredictions(
            job_name=self.job_name,
            external_id_array=[asset.externalId for asset in assets],
            model_name_array=[self.model_name] * len(assets),
            json_response_array=[
                {self.job_name: self._create_categories(prob_array)} for prob_array in prob_arrays
            ],
            predictions_probability=np.max(np.array(prob_arrays), axis=1).tolist(),
        )
        return job_predictions

    def _create_categories(self, prob_array) -> JsonResponseClassification:
        return {
            "categories": [
                {
                    "name": list(self.class_name_to_idx.keys())[np.argmax(prob_array)],
                    "confidence": int(np.max(prob_array) * 100),
                }
            ]
        }

    def get_model_path(self, model_path, from_project):
        model_name: ModelNameT = self.model_name  # type: ignore

        if model_path is not None:
            model_path_set = model_path
        elif from_project is not None:
            model_path_repository_dir = Path.model_repository_dir(
                from_project, self.job_name, self.model_conditions.model_repository
            )

            model_path_from_project = PathPytorchVision.append_model_path(
                model_path_repository_dir, model_name
            )
            model_path_set = model_path_from_project
        else:
            model_path_set = self.model_path
        return model_path_set

    def find_errors(
        self,
        *,
        assets: AssetsLazyList,
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool = False,
        api_key: str = "",
    ):
        _ = clear_dataset_cache

        images = download_project_images(
            api_key=api_key, assets=assets, output_folder=self.data_dir
        )
        labels = []
        for asset in assets:
            labels.append(
                asset.get_annotations_classification(self.job_name)["categories"][0]["name"]
            )

        kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=42)
        probability_matrix = np.empty((len(labels), len(self.class_name_to_idx)))

        for cv_fold in tqdm(range(cv_n_folds), desc="Training and predicting on several folds"):
            # Split train into train and holdout for particular cv_fold.
            cv_train_idx, cv_holdout_idx = list(kf.split(range(len(labels)), labels))[cv_fold]
            splits = {}
            splits["train"], splits["val"] = train_test_split(
                cv_train_idx, test_size=0.2, random_state=42
            )
            image_datasets = {
                x: ClassificationTrainDataset(
                    [images[i] for i in splits[x]],
                    [labels[i] for i in splits[x]],
                    self.class_name_to_idx,
                    data_transforms[x],
                )
                for x in ["train", "val"]
            }
            holdout_dataset = ClassificationPredictDataset(
                [images[i] for i in cv_holdout_idx],
                data_transforms["val"],
            )
            logger.debug(f"\nCV Fold: {cv_fold+1}/{cv_n_folds}")
            logger.debug(f"Train size: {len(image_datasets['train'])}")
            logger.debug(f"Validation size: {len(image_datasets['val'])}")
            logger.debug(f"Holdout size: {len(holdout_dataset)}")

            model_name: ModelNameT = self.model_name  # type: ignore

            model, _ = get_trained_model_image_classif(
                model_name=model_name,
                batch_size=batch_size,
                category_ids=self.class_names,
                epochs=epochs,
                image_datasets=image_datasets,
                save_model_path=None,
                disable_wandb=True,
            )

            holdout_loader = torch_Data.DataLoader(
                holdout_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )

            probs = predict_probabilities(holdout_loader, model)
            probability_matrix[cv_holdout_idx] = probs

        destination = os.path.join(self.model_dir, "train_model_intel_probability_matrix.npy")
        np.save(destination, probability_matrix)

        labels_idx = [self.class_name_to_idx[label] for label in labels]
        noise_indices = find_label_issues(
            labels_idx, probability_matrix, return_indices_ranked_by="normalized_margin"
        )

        noise_paths = []
        for idx in noise_indices:
            noise_paths.append(images[idx].id)

        return ErrorRecap(
            id_array=noise_paths,
            external_id_array=[asset.externalId for asset in assets],
            errors_by_asset=[
                [
                    LabelingError(error_type="misclassification", error_probability=0.4)
                ]  # TODO: use true proba
                for _ in noise_paths
            ],
        )
