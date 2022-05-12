import copy
import os
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
from kiliautoml.utils.constants import HOME, ModelNameT, ModelRepositoryT
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


class PyTorchVisionImageClassificationModel(BaseModel):
    def __init__(
        self,
        *,
        project_id: str,
        api_key: str,
        assets: List[Any],
        job_name: str,
        model_repository: Optional[ModelRepositoryT],
        model_name: Optional[ModelNameT],
    ):

        model_repository = set_model_repository_image_classification(model_repository)
        model_name = set_model_name_image_classification(model_name)
        model_repository_dir = Path.model_repository_dir(
            HOME, project_id, job_name, model_repository
        )

        model_dir = PathPytorchVision.append_model_dir(model_repository_dir)
        model_path = PathPytorchVision.append_model_path(model_repository_dir, model_name)
        data_dir = os.path.join(model_repository_dir, "data")
        download_project_image_clean_lab(
            assets=assets,
            api_key=api_key,
            data_path=data_dir,
            job_name=job_name,
            project_id=project_id,
        )

        # To set to False if the input size varies a lot and you see that the training takes
        # too much time
        cudnn.benchmark = True

        original_image_datasets = get_original_image_dataset(data_dir)

        class_names = original_image_datasets["train"].classes  # type: ignore
        labels = [label for img, label in datasets.ImageFolder(data_dir).imgs]
        assert len(class_names) > 1, "There should be at least 2 classes in the dataset."

        self.model_name: ModelNameT = model_name
        self.model_dir = model_dir
        self.model_path = model_path
        self.data_dir = data_dir
        self.original_image_datasets = original_image_datasets
        self.class_names = class_names
        self.labels = labels

    def train(self, epochs, verbose):

        image_datasets = copy.deepcopy(self.original_image_datasets)
        train_idx, val_idx = train_test_split(
            range(len(self.labels)), test_size=0.2, random_state=42
        )
        prepare_image_dataset(train_idx, val_idx, image_datasets)
        model, loss = get_trained_model_image_classif(
            epochs,
            self.model_name,
            verbose,
            self.class_names,
            image_datasets,
            save_model_path=self.model_path,
        )
        return loss

    def predict(self, verbose: int, assets, job_name):
        model = initialize_model_img_class(self.model_name, self.class_names)
        model.load_state_dict(torch.load(self.model_path))
        loader = torch_Data.DataLoader(
            self.original_image_datasets["val"], batch_size=1, shuffle=False, num_workers=1
        )
        probs = predict_probabilities(loader, model, verbose=verbose)

        job_predictions = JobPredictions(
            job_name=job_name,
            external_id_array=[asset["externalId"] for asset in assets],
            model_name_array=[self.model_name] * len(self.labels),
            json_response_array=[asset["labels"][0]["jsonResponse"] for asset in assets],
            predictions_probability=list(np.max(probs, axis=1)),
        )
        return job_predictions

    def find_errors(self, cv_n_folds, epochs, verbose):
        kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=42)
        for cv_fold in tqdm(range(cv_n_folds)):  # type: ignore
            # Split train into train and holdout for particular cv_fold.
            cv_train_idx, cv_holdout_idx = list(kf.split(range(len(self.labels)), self.labels))[
                cv_fold
            ]

            image_datasets, holdout_dataset = separe_holdout_datasets(
                cv_n_folds,
                verbose,
                self.original_image_datasets,
                cv_fold,
                cv_train_idx,
                cv_holdout_idx,
            )

            model, loss = get_trained_model_image_classif(
                epochs,
                self.model_name,
                verbose,
                self.class_names,
                image_datasets,
                save_model_path=None,
            )

            holdout_loader = torch_Data.DataLoader(
                holdout_dataset,
                batch_size=64,
                shuffle=False,
                pin_memory=True,
            )

            probs = predict_probabilities(holdout_loader, model, verbose=verbose)
            print("probs", probs)
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

        noise_indices = find_label_issues(
            self.labels, psx, return_indices_ranked_by="normalized_margin"
        )

        noise_paths = []
        for idx in noise_indices:
            noise_paths.append(os.path.basename(train_imgs[idx][0])[:-4])
        return noise_paths
