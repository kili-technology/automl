import copy
import os
import time
from typing import Any, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_Data
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from kiliautoml.utils.constants import HOME, ModelName, ModelNameT, ModelRepositoryT
from kiliautoml.utils.download_assets import download_project_image_clean_lab
from kiliautoml.utils.helpers import set_default
from kiliautoml.utils.path import Path, PathPytorchVision


def train_model_pytorch(
    *,
    model,
    dataloaders,
    verbose=0,
    epochs=10,
):
    """
    Method that trains the given model and return the best one found in the given epochs
    """
    since = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = model.to(device)  # type:ignore
    dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        if verbose >= 2:
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]  # type:ignore

            if verbose >= 2:
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        if verbose >= 2:
            print()

    if verbose >= 2:
        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_probs(loader, model, verbose=0):
    """
    Method to compute the probabilities for all classes for the assets in the holdout set
    """
    # Switch to evaluate mode.
    model.eval()
    n_total = len(loader.dataset.imgs) / float(loader.batch_size)
    outputs = []
    if verbose >= 2:
        print("Computing probabilities for this fold")
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            if verbose >= 2:
                print("\rComplete: {:.1%}".format(i / n_total), end="")
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            outputs.append(model(input))
        if verbose >= 2:
            print()

    # Prepare outputs as a single matrix
    probs = np.concatenate(
        [torch.nn.functional.softmax(z, dim=1).cpu().numpy() for z in outputs]  # type:ignore
    )

    return probs


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
    model_repository = set_default(
        model_repository,
        "torchvision",
        "model_repository",
        ["torchvision"],
    )

    model_name = set_default(  # type:ignore
        model_name,
        ModelName.EfficientNetB0,
        "model_name",
        [ModelName.EfficientNetB0, ModelName.Resnet50],
    )

    model_repository_path = Path.model_repository(HOME, project_id, job_name, model_repository)

    # TODO: move to Path
    model_dir = PathPytorchVision.append_model_folder(model_repository_path)
    data_dir = os.path.join(model_repository_path, "data")

    download_project_image_clean_lab(assets, api_key, data_dir, job_name)

    # To set to False if the input size varies a lot and you see that the training takes
    # too much time
    cudnn.benchmark = True

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

    class_names = original_image_datasets["train"].classes
    labels = [label for img, label in datasets.ImageFolder(data_dir).imgs]
    for cv_fold in range(cv_n_folds):
        # Split train into train and holdout for particular cv_fold.
        kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=cv_seed)
        cv_train_idx, cv_holdout_idx = list(kf.split(range(len(labels)), labels))[cv_fold]
        # Separate datasets
        np.random.seed(cv_seed)
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

        dataloaders = {
            x: torch_Data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4)
            for x in ["train", "val"]
        }

        if model_name == ModelName.EfficientNetB0:
            model = models.efficientnet_b0(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, len(class_names))  # type:ignore
        elif model_name == ModelName.Resnet50:
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(class_names))
        else:
            raise ValueError(f"Model {model_name} not supported.")

        model = train_model_pytorch(
            model=model,
            dataloaders=dataloaders,
            verbose=verbose,
            epochs=epochs,
        )
        # torch.save(model_ft.state_dict(), os.path.join(model_dir, f'model_{cv_fold}.pt'))

        holdout_loader = torch_Data.DataLoader(
            holdout_dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
        )

        probs = get_probs(holdout_loader, model, verbose=verbose)
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
