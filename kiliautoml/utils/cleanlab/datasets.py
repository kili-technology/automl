import copy
from typing import Dict

import torch.utils.data as torch_Data
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from typing_extensions import Literal

training_phaseT = Literal["train", "test"]
dict_datasetT = Dict[training_phaseT, torch_Data.Dataset]


def get_original_image_dataset(data_dir) -> dict_datasetT:
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

    return original_image_datasets  # type: ignore


def prepare_image_dataset(train_idx, val_idx, image_datasets):
    image_datasets["train"].imgs = [image_datasets["train"].imgs[i] for i in train_idx]
    image_datasets["train"].samples = image_datasets["train"].imgs
    image_datasets["val"].imgs = [image_datasets["val"].imgs[i] for i in val_idx]
    image_datasets["val"].samples = image_datasets["val"].imgs


def separe_holdout_datasets(
    cv_n_folds, verbose, original_image_datasets, cv_fold, cv_train_idx, cv_holdout_idx
):
    train_idx, val_idx = train_test_split(cv_train_idx, test_size=0.2)
    image_datasets = copy.deepcopy(original_image_datasets)
    holdout_dataset = copy.deepcopy(image_datasets["val"])
    holdout_dataset.imgs = [holdout_dataset.imgs[i] for i in cv_holdout_idx]
    holdout_dataset.samples = holdout_dataset.imgs
    prepare_image_dataset(train_idx, val_idx, image_datasets)

    if verbose >= 1:
        print(f"\nCV Fold: {cv_fold+1}/{cv_n_folds}")
        print(f"Train size: {len(image_datasets['train'].imgs)}")
        print(f"Validation size: {len(image_datasets['val'].imgs)}")
        print(f"Holdout size: {len(holdout_dataset.imgs)}")
        print()
    return image_datasets, holdout_dataset
