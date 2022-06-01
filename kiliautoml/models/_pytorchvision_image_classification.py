import os
from typing import Any, List, Optional
from warnings import warn

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as torch_Data
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import datasets
from tqdm import tqdm

from kiliautoml.models._base_model import BaseModel
from kiliautoml.utils.cleanlab.train_cleanlab import combine_folds
from kiliautoml.utils.constants import (
    HOME,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
)
from kiliautoml.utils.download_assets import download_project_images
from kiliautoml.utils.helpers import JobPredictions, get_label
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
from kiliautoml.utils.type import AssetT, JobT, LabelMergeT


class PyTorchVisionImageClassificationModel(BaseModel):
    def __init__(
        self,
        *,
        project_id: str,
        model_repository: Optional[ModelRepositoryT],
        job: JobT,
        job_name: str,
        model_name: ModelNameT,
        model_framework: ModelFrameworkT,
    ):
        model_repository = set_model_repository_image_classification(model_repository)
        model_name = set_model_name_image_classification(model_name)
        model_repository_dir = Path.model_repository_dir(
            HOME, project_id, job_name, model_repository
        )

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

        self.class_name_to_idx = {}
        for i, category in enumerate(job["content"]["categories"]):
            self.class_name_to_idx[job["content"]["categories"][category]["name"]] = i
        self.class_names = list(self.class_name_to_idx.keys())

    def train(
        self,
        *,
        assets: List[AssetT],
        label_merge: LabelMergeT,
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
        disable_wandb: bool,
        verbose: int = 1,
        api_key: str = "",
    ):
        _ = clear_dataset_cache

        if disable_wandb is False:
            warn("Wandb is not supported for this model.")

        images = download_project_images(
            api_key=api_key, assets=assets, output_folder=self.data_dir
        )
        labels = []
        for asset in assets:
            kili_label = get_label(asset, label_merge)
            if (kili_label is None) or (self.job_name not in kili_label["jsonResponse"]):
                asset_id = asset["id"]
                warn(f"${asset_id}: No annotation for job ${self.job_name}")
                return 0.0
            else:
                labels.append(kili_label["jsonResponse"][self.job_name]["categories"][0]["name"])

        split = {}
        split["train"], split["val"] = train_test_split(
            range(len(labels)), test_size=0.2, random_state=42
        )

        image_datasets = {
            x: ClassificationTrainDataset(
                [images[i] for i in split[x]],
                [labels[i] for i in split[x]],
                self.class_name_to_idx,
                data_transforms[x],
            )
            for x in ["train", "val"]
        }

        _, loss = get_trained_model_image_classif(
            epochs=epochs,
            model_name=self.model_name,  # type: ignore
            batch_size=batch_size,
            verbose=verbose,
            class_names=self.class_names,
            image_datasets=image_datasets,
            save_model_path=self.model_path,
        )
        return loss

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
        probs = predict_probabilities(loader, model, verbose=verbose)

        job_predictions = JobPredictions(
            job_name=self.job_name,
            external_id_array=[asset["externalId"] for asset in assets],
            model_name_array=[self.model_name] * len(assets),
            json_response_array=[
                {
                    "CLASSIFICATION_JOB": {
                        "categories": [
                            {"name": list(self.class_name_to_idx.keys())[np.argmax(prob)]}
                        ]
                    }
                }
                for prob in probs
            ],
            predictions_probability=list(np.max(probs, axis=1)),
        )
        return job_predictions

    def get_model_path(self, model_path, from_project):
        model_name: ModelNameT = self.model_name  # type: ignore

        if model_path is not None:
            model_path_set = model_path
        elif from_project is not None:
            model_path_repository_dir = Path.model_repository_dir(
                HOME, from_project, self.job_name, self.model_repository
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
        label_merge: LabelMergeT,
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
            kili_label = get_label(asset, label_merge)
            if (kili_label is None) or (self.job_name not in kili_label["jsonResponse"]):
                asset_id = asset["id"]
                warn(f"${asset_id}: No annotation for job ${self.job_name}")
                return 0.0
            else:
                labels.append(kili_label["jsonResponse"][self.job_name]["categories"][0]["name"])

        kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=42)
        for cv_fold in tqdm(range(cv_n_folds), desc="Training and predicting on several folds"):
            # Split train into train and holdout for particular cv_fold.
            cv_train_idx, cv_holdout_idx = list(kf.split(range(len(labels)), labels))[cv_fold]
            split = {}
            split["train"], split["val"] = train_test_split(
                cv_train_idx, test_size=0.2, random_state=42
            )
            image_datasets = {
                x: ClassificationTrainDataset(
                    [images[i] for i in split[x]],
                    [labels[i] for i in split[x]],
                    self.class_name_to_idx,
                    data_transforms[x],
                )
                for x in ["train", "val"]
            }
            holdout_dataset = ClassificationTrainDataset(
                [images[i] for i in cv_holdout_idx],
                [labels[i] for i in cv_holdout_idx],
                self.class_name_to_idx,
                data_transforms["val"],
            )
            if verbose >= 1:
                print(f"\nCV Fold: {cv_fold+1}/{cv_n_folds}")
                print(f"Train size: {len(image_datasets['train'])}")
                print(f"Validation size: {len(image_datasets['val'])}")
                print(f"Holdout size: {len(holdout_dataset)}")
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
            probs_path = os.path.join(self.model_dir, "model_fold_{}__probs.npy".format(cv_fold))
            np.save(probs_path, probs)

        psx_path = combine_folds(
            data_dir=self.data_dir,
            model_dir=self.model_dir,
            num_classes=len(self.class_names),
            verbose=verbose,
        )

        psx = np.load(psx_path)
        train_imgs = datasets.ImageFolder(self.data_dir).imgs

        noise_indices = find_label_issues(labels, psx, return_indices_ranked_by="normalized_margin")

        noise_paths = []
        for idx in noise_indices:
            noise_paths.append(os.path.basename(train_imgs[idx][0])[:-4])
        return noise_paths
