import os
import warnings
from typing import Any, List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as torch_Data
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.autonotebook import tqdm

from kiliautoml.models._base_model import BaseModel
from kiliautoml.utils.download_assets import download_project_images
from kiliautoml.utils.helpers import JobPredictions, kili_print
from kiliautoml.utils.path import Path, PathPytorchVision
from kiliautoml.utils.pytorchvision.image_classification import (
    ClassificationPredictDataset,
    ClassificationTrainDataset,
    data_transforms,
    get_trained_model_image_classif,
    initialize_model_img_class,
    predict_probabilities,
    set_model_name_image_classification,
    set_model_repository_image_classification,
)
from kiliautoml.utils.type import (
    AssetT,
    JobNameT,
    JobT,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
)


class PyTorchVisionImageClassificationModel(BaseModel):
    def __init__(
        self,
        *,
        project_id: ProjectIdT,
        model_repository: Optional[ModelRepositoryT],
        job: JobT,
        job_name: JobNameT,
        model_name: ModelNameT,
        model_framework: ModelFrameworkT,
    ):
        model_repository = set_model_repository_image_classification(model_repository)
        model_name = set_model_name_image_classification(model_name)
        model_repository_dir = Path.model_repository_dir(project_id, job_name, model_repository)

        model_dir = PathPytorchVision.append_model_dir(model_repository_dir)
        model_path = PathPytorchVision.append_model_path(model_repository_dir, model_name)
        data_dir = PathPytorchVision.append_data_dir(model_repository_dir)

        # To set to False if the input size varies a lot and you see that the training takes
        # too much time
        cudnn.benchmark = True

        BaseModel.__init__(
            self,
            job=job,
            job_name=job_name,
            model_name=model_name,
            model_framework=model_framework,
        )

        self.model_dir = model_dir
        self.model_path = model_path
        self.data_dir = data_dir

        self.class_name_to_idx = {
            category: i for i, category in enumerate(job["content"]["categories"])
        }
        self.class_names = list(self.class_name_to_idx.keys())

    def train(
        self,
        *,
        assets: List[AssetT],
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
        disable_wandb: bool,
        verbose: int = 1,
        api_key: str = "",
    ):
        _ = clear_dataset_cache

        if disable_wandb is False:
            warnings.warn("Wandb is not supported for this model.")

        images = download_project_images(
            api_key=api_key, assets=assets, output_folder=self.data_dir
        )
        labels = []
        for asset in assets:
            labels.append(
                asset["labels"][0]["jsonResponse"][self.job_name]["categories"][0]["name"]
            )

        splits = {}
        splits["train"], splits["val"] = train_test_split(
            range(len(labels)), test_size=0.2, random_state=42
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
            verbose=verbose,
            class_names=self.class_names,
            image_datasets=image_datasets,
            save_model_path=self.model_path,
        )
        return model_evaluation

    def predict(
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
        _ = clear_dataset_cache

        images = download_project_images(
            api_key=api_key, assets=assets, output_folder=self.data_dir
        )

        dataset = ClassificationPredictDataset(images, data_transforms["val"])

        model = initialize_model_img_class(
            self.model_name,  # type: ignore
            self.class_names,
        )

        model_path = self.get_model_path(model_path, from_project)

        model.load_state_dict(torch.load(model_path))
        loader = torch_Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        prob_arrays = predict_probabilities(loader, model, verbose=verbose)

        job_predictions = JobPredictions(
            job_name=self.job_name,
            external_id_array=[asset["externalId"] for asset in assets],
            model_name_array=[self.model_name] * len(assets),
            json_response_array=[
                {
                    "CLASSIFICATION_JOB": {  # TODO: replace by self.job_name
                        "categories": [
                            {
                                "name": list(self.class_name_to_idx.keys())[np.argmax(prob_array)],
                                "confidence": np.max(prob_array),
                            }
                        ]
                    }
                }
                for prob_array in prob_arrays
            ],
            predictions_probability=prob_arrays,
        )
        return job_predictions

    def get_model_path(self, model_path, from_project):
        model_name: ModelNameT = self.model_name  # type: ignore

        if model_path is not None:
            model_path_set = model_path
        elif from_project is not None:
            model_path_repository_dir = Path.model_repository_dir(
                from_project, self.job_name, self.model_repository
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
        assets: List[AssetT],
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: int = 0,
        clear_dataset_cache: bool = False,
        api_key: str = "",
    ) -> Any:
        _ = clear_dataset_cache

        images = download_project_images(
            api_key=api_key, assets=assets, output_folder=self.data_dir
        )
        labels = []
        for asset in assets:
            labels.append(
                asset["labels"][0]["jsonResponse"][self.job_name]["categories"][0]["name"]
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
            if verbose >= 1:
                kili_print(f"\nCV Fold: {cv_fold+1}/{cv_n_folds}")
                kili_print(f"Train size: {len(image_datasets['train'])}")
                kili_print(f"Validation size: {len(image_datasets['val'])}")
                kili_print(f"Holdout size: {len(holdout_dataset)}")
                print()

            model_name: ModelNameT = self.model_name  # type: ignore

            model, _ = get_trained_model_image_classif(
                model_name=model_name,
                batch_size=batch_size,
                verbose=verbose,
                class_names=self.class_names,
                epochs=epochs,
                image_datasets=image_datasets,
                save_model_path=None,
            )

            holdout_loader = torch_Data.DataLoader(
                holdout_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )

            probs = predict_probabilities(holdout_loader, model, verbose=verbose)
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
        return noise_paths
