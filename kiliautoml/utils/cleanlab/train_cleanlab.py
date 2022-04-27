import copy
import os
import time

from cleanlab.filter import find_label_issues
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms

from kiliautoml.utils.constants import ModelName


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    dataset_sizes,
    verbose=0,
    num_epochs=10,
):
    """
    Method that trains the given model and return the best one found in the given epochs
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if verbose >= 2:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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
    probs = np.concatenate([torch.nn.functional.softmax(z, dim=1).cpu().numpy() for z in outputs])

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
        probs_path = os.path.join(model_dir, "model_fold_{}__probs.npy".format(k))
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


def train_and_get_error_labels(
    cv_n_folds,
    data_dir,
    epochs,
    model_dir,
    model_name,
    verbose=0,
    cv_seed=42,
):
    """
    Main method that trains the model on the assets that are in data_dir, computes the
    incorrect labels using Cleanlab and returns the IDs of the concerned assets.
    """
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for cv_fold in range(cv_n_folds):
        labels = [label for img, label in datasets.ImageFolder(data_dir).imgs]
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
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=64, shuffle=True, num_workers=4
            )
            for x in ["train", "val"]
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
        class_names = image_datasets["train"].classes

        if model_name == ModelName.EfficientNetB0:
            model_ft = models.efficientnet_b0(pretrained=True)
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))
        elif model_name == ModelName.Resnet50:
            model_ft = models.resnet50(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, len(class_names))

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft = train_model(
            model_ft,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            dataloaders,
            device,
            dataset_sizes,
            verbose=verbose,
            num_epochs=epochs,
        )
        # torch.save(model_ft.state_dict(), os.path.join(model_dir, f'model_{cv_fold}.pt'))

        holdout_loader = torch.utils.data.DataLoader(
            holdout_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        probs = get_probs(holdout_loader, model_ft, verbose=verbose)
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
