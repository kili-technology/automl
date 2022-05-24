import copy
import os
import shutil
from typing import Any, List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as torch_Data
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import datasets
from tqdm import tqdm

from kiliautoml.models._base_model import BaseModel
from kiliautoml.utils.cleanlab.datasets import (
    get_original_image_dataset,
    prepare_image_dataset,
    separe_holdout_datasets,
)
from kiliautoml.utils.cleanlab.train_cleanlab import combine_folds
from kiliautoml.utils.constants import (
    HOME,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
)
from kiliautoml.utils.download_assets import download_project_image_clean_lab
from kiliautoml.utils.helpers import JobPredictions
from kiliautoml.utils.path import Path, PathPytorchVision
from kiliautoml.utils.pytorchvision.image_classification import (
    get_trained_model_image_classif,
    initialize_model_img_class,
    predict_probabilities,
    set_model_name_image_classification,
    set_model_repository_image_classification,
)
from kiliautoml.utils.type import AssetT, JobT


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

    def prepare_dataset(
        self,
        assets: List[AssetT],
        api_key,
    ):
        shutil.rmtree(self.data_dir, ignore_errors=True)
        download_project_image_clean_lab(
            assets=assets,
            api_key=api_key,
            data_path=self.data_dir,
            job_name=self.job_name,
        )
        original_image_datasets = get_original_image_dataset(self.data_dir)

        class_names = original_image_datasets["train"].classes  # type: ignore
        labels = [label for _, label in datasets.ImageFolder(self.data_dir).imgs]
        assert len(class_names) > 1, "There should be at least 2 classes in the dataset."
        return original_image_datasets, labels, class_names

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
            raise NotImplementedError("Wandb is not supported for this model.")
        original_image_datasets, labels, class_names = self.prepare_dataset(assets, api_key)

        image_datasets = copy.deepcopy(original_image_datasets)
        train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)
        prepare_image_dataset(train_idx, val_idx, image_datasets)
        _, loss = get_trained_model_image_classif(
            epochs=epochs,
            model_name=self.model_name,  # type: ignore
            batch_size=batch_size,
            verbose=verbose,
            class_names=class_names,
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

        original_image_datasets, labels, class_names = self.prepare_dataset(assets, api_key)
        _ = labels

        model = initialize_model_img_class(
            self.model_name,  # type: ignore
            class_names,
        )

        model_path = self.get_model_path(model_path, from_project)

        model.load_state_dict(torch.load(model_path))
        loader = torch_Data.DataLoader(
            original_image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=1
        )
        probs = predict_probabilities(loader, model, verbose=verbose)

        job_predictions = JobPredictions(
            job_name=self.job_name,
            external_id_array=[asset["externalId"] for asset in assets],
            model_name_array=[self.model_name] * len(assets),
            json_response_array=[asset["labels"][0]["jsonResponse"] for asset in assets],
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
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: int = 0,
        clear_dataset_cache: bool = False,
        api_key: str = "",
    ) -> Any:
        _ = clear_dataset_cache

        original_image_datasets, labels, class_names = self.prepare_dataset(assets, api_key)
        kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=42)
        for cv_fold in tqdm(range(cv_n_folds)):
            # Split train into train and holdout for particular cv_fold.
            cv_train_idx, cv_holdout_idx = list(kf.split(range(len(labels)), labels))[cv_fold]

            image_datasets, holdout_dataset = separe_holdout_datasets(
                cv_n_folds,
                verbose,
                original_image_datasets,
                cv_fold,
                cv_train_idx,
                cv_holdout_idx,
            )
            model_name: ModelNameT = self.model_name  # type: ignore

            model, _ = get_trained_model_image_classif(
                model_name=model_name,
                batch_size=batch_size,
                verbose=verbose,
                class_names=class_names,
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
            num_classes=len(class_names),
            verbose=verbose,
        )

        psx = np.load(psx_path)
        train_imgs = datasets.ImageFolder(self.data_dir).imgs

        noise_indices = find_label_issues(labels, psx, return_indices_ranked_by="normalized_margin")

        noise_paths = []
        for idx in noise_indices:
            noise_paths.append(os.path.basename(train_imgs[idx][0])[:-4])
        return noise_paths
