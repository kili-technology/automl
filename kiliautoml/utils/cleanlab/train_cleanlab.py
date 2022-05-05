import copy
import os
from typing import Any, List

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data as torch_Data
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import datasets, transforms

from kiliautoml.utils.constants import HOME, ModelNameT, ModelRepositoryT
from kiliautoml.utils.download_assets import download_project_image_clean_lab
from kiliautoml.utils.path import Path, PathPytorchVision
from kiliautoml.utils.pytorchvision.image_classification import (
    get_trained_model_image_classif,
    predict_probabilities,
    set_model_name_image_classification,
    set_model_repository_image_classification,
)


def combine_folds(data_dir, model_dir, verbose=0, num_classes=10, nb_folds=4, seed=42):
    """
    Method that combines the probabilities from all the holdout sets into a single file
    """
    destination = os.path.join(model_dir, "train_model_intel_pyx.npy")
    if verbose >= 2:
        print()
        print("Combining probabilities. This method will overwrite file: {}".format(destination))
    # Prepare labels
    labels = [label for _, label in datasets.ImageFolder(data_dir).imgs]
    # Initialize pyx array (output of trained network)
    pyx = np.empty((len(labels), num_classes))

    # Split train into train and holdout for each cv_fold.
    kf = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=seed)
    for k, (_, cv_holdout_idx) in enumerate(kf.split(range(len(labels)), labels)):
        probs_path = os.path.join(model_dir, f"model_fold_{k}__probs.npy")
        probs = np.load(probs_path)
        pyx[cv_holdout_idx] = probs[:, :num_classes]
    if verbose >= 2:
        print("Writing final predicted probabilities.")
    np.save(destination, pyx)

    if verbose >= 2:
        # Compute overall accuracy
        print("Computing Accuracy.", flush=True)
        acc = sum(np.array(labels) == np.argmax(pyx, axis=1)) / float(len(labels))
        print("Accuracy: {:.25}".format(acc))

    return destination


def train_and_get_error_image_classification(
    cv_n_folds: int,
    epochs: int,
    assets: List[Any],
    model_repository: ModelRepositoryT,
    model_name: ModelNameT,
    job_name: str,
    project_id: str,
    api_key: str,
    verbose: int = 0,
    cv_seed: int = 42,
):
    """
    Main method that trains the model on the assets that are in data_dir, computes the
    incorrect labels using Cleanlab and returns the IDs of the concerned assets.
    """
    model_repository = set_model_repository_image_classification(model_repository)
    model_name = set_model_name_image_classification(model_name)
    model_repository_path = Path.model_repository(HOME, project_id, job_name, model_repository)

    # TODO: move to Path
    model_dir = PathPytorchVision.append_model_folder(model_repository_path)
    data_dir = os.path.join(model_repository_path, "data")
    download_project_image_clean_lab(assets, api_key, data_dir, job_name)

    # To set to False if the input size varies a lot and you see that the training takes
    # too much time
    cudnn.benchmark = True

    original_image_datasets = get_original_image_dataset(data_dir)

    class_names = original_image_datasets["train"].classes
    labels = [label for img, label in datasets.ImageFolder(data_dir).imgs]
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=cv_seed)
    for cv_fold in range(cv_n_folds):
        # Split train into train and holdout for particular cv_fold.
        cv_train_idx, cv_holdout_idx = list(kf.split(range(len(labels)), labels))[cv_fold]

        # Separate datasets
        image_datasets, holdout_dataset = separe_holdout_datasets(
            cv_n_folds, verbose, original_image_datasets, cv_fold, cv_train_idx, cv_holdout_idx
        )

        model = get_trained_model_image_classif(
            epochs, model_name, verbose, class_names, image_datasets
        )
        # torch.save(model.state_dict(), os.path.join(model_dir, f'model_{cv_fold}.pt'))

        holdout_loader = torch_Data.DataLoader(
            holdout_dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
        )
        probs = predict_probabilities(holdout_loader, model, verbose=verbose)

        probs_path = os.path.join(model_dir, "model_fold_{}__probs.npy".format(cv_fold))
        np.save(probs_path, probs)

    psx_path = combine_folds(
        data_dir=data_dir,
        model_dir=model_dir,
        num_classes=len(class_names),
        seed=cv_seed,
        verbose=verbose,
    )

    psx = np.load(psx_path)
    train_imgs = datasets.ImageFolder(data_dir).imgs

    noise_indices = find_label_issues(labels, psx, return_indices_ranked_by="normalized_margin")

    noise_paths = []
    for idx in noise_indices:
        noise_paths.append(os.path.basename(train_imgs[idx][0])[:-4])

    return noise_paths


def get_original_image_dataset(data_dir):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    }

    original_image_datasets = {
        x: datasets.ImageFolder(data_dir, data_transforms[x]) for x in ["train", "val"]
    }

    return original_image_datasets


def separe_holdout_datasets(
    cv_n_folds, verbose, original_image_datasets, cv_fold, cv_train_idx, cv_holdout_idx
):
    train_idx, val_idx = train_test_split(cv_train_idx, test_size=0.2)
    image_datasets = copy.deepcopy(original_image_datasets)
    holdout_dataset = copy.deepcopy(image_datasets["val"])
    holdout_dataset.imgs = [holdout_dataset.imgs[i] for i in cv_holdout_idx]
    holdout_dataset.samples = holdout_dataset.imgs
    image_datasets["train"].imgs = [image_datasets["train"].imgs[i] for i in train_idx]
    image_datasets["train"].samples = image_datasets["train"].imgs
    image_datasets["val"].imgs = [image_datasets["val"].imgs[i] for i in val_idx]
    image_datasets["val"].samples = image_datasets["val"].imgs

    if verbose >= 1:
        print(f"\nCV Fold: {cv_fold+1}/{cv_n_folds}")
        print(f"Train size: {len(image_datasets['train'].imgs)}")
        print(f"Validation size: {len(image_datasets['val'].imgs)}")
        print(f"Holdout size: {len(holdout_dataset.imgs)}")
        print()
    return image_datasets, holdout_dataset
