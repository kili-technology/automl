import copy
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import lr_scheduler
from tqdm.autonotebook import trange

from kiliautoml.utils.helpers import kili_print
from kiliautoml.utils.type import Model_Metric

# Necessary on mac for train and predict.
os.environ["OMP_NUM_THREADS"] = "1"


def train_model_pytorch(
    *,
    model: nn.Module,
    dataloaders,
    epochs,
    verbose=0,
    class_names,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Method that trains the given model and return the best one found in the given epochs
    """
    since = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose >= 2:
        kili_print("Start model training on device: {}".format(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = model.to(device)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_metrics = {}
    best_val_metrics["loss"] = Model_Metric(overall=float("inf"), by_category=None)
    epoch_train_evaluation = {}
    corresponding_train_metrics = {}
    for _ in trange(epochs, desc="Training - Epoch"):
        if verbose >= 2:
            print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            ys_pred = []
            ys_true = []
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
                ys_pred.append(preds.cpu())
                ys_true.append(labels.cpu())
            if phase == "train":
                scheduler.step()
                epoch_train_evaluation = evaluate(
                    running_loss, np.concatenate(ys_pred), np.concatenate(ys_true)
                )
                epoch_train_loss = epoch_train_evaluation["loss"]["overall"]
                epoch_train_acc = epoch_train_evaluation["acc"]["overall"]
                if verbose >= 2:
                    print(f"{phase} Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")
            if phase == "val":
                epoch_val_evaluation = evaluate(
                    running_loss, np.concatenate(ys_pred), np.concatenate(ys_true)
                )
                epoch_val_loss = epoch_val_evaluation["loss"]["overall"]
                epoch_val_acc = epoch_val_evaluation["acc"]["overall"]
                if verbose >= 2:
                    print(f"{phase} Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
                # deep copy the model
                if epoch_val_loss < best_val_metrics["loss"]["overall"]:
                    best_val_metrics = epoch_val_evaluation
                    corresponding_train_metrics = epoch_train_evaluation
                    best_model_wts = copy.deepcopy(model.state_dict())
        if verbose >= 2:
            print()

    if verbose >= 2:
        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        best_val_loss = best_val_metrics["loss"]["overall"]
        best_val_acc = best_val_metrics["acc"]["overall"]
        corresponding_train_loss = corresponding_train_metrics["loss"]["overall"]
        corresponding_train_acc = corresponding_train_metrics["acc"]["overall"]
        print(f"Best val Loss: {best_val_loss:4f}, Best val Acc: {best_val_acc:4f}")
        print(
            f"Corresponding train Loss: {corresponding_train_loss:4f},"
            f"Best val Acc: {corresponding_train_acc:4f}"
        )

    # load best model weights
    model.load_state_dict(best_model_wts)
    model_evaluation: Dict[str, Any] = {}

    for i, label in enumerate(class_names):
        model_evaluation["train_" + label] = {}
        model_evaluation["val_" + label] = {}
        model_evaluation["train_" + label]["precision"] = corresponding_train_metrics["precision"][
            "by_category"
        ][i]
        model_evaluation["train_" + label]["recall"] = corresponding_train_metrics["recall"][
            "by_category"
        ][i]
        model_evaluation["train_" + label]["f1"] = corresponding_train_metrics["f1"]["by_category"][
            i
        ]
        model_evaluation["val_" + label]["precision"] = best_val_metrics["precision"][
            "by_category"
        ][  # type:ignore
            i
        ]
        model_evaluation["val_" + label]["recall"] = best_val_metrics["recall"][
            "by_category"
        ][  # type:ignore
            i
        ]
        model_evaluation["val_" + label]["f1"] = best_val_metrics["f1"][
            "by_category"
        ][  # type:ignore
            i
        ]

    model_evaluation["train__overall"] = {
        "loss": corresponding_train_metrics["loss"]["overall"],
        "accuracy": corresponding_train_metrics["acc"]["overall"],
        "precision": corresponding_train_metrics["precision"]["overall"],
        "recall": corresponding_train_metrics["recall"]["overall"],
        "f1": corresponding_train_metrics["f1"]["overall"],
    }

    model_evaluation["val__overall"] = {
        "loss": best_val_metrics["loss"]["overall"],
        "accuracy": best_val_metrics["acc"]["overall"],  # type:ignore
        "precision": best_val_metrics["precision"]["overall"],  # type:ignore
        "recall": best_val_metrics["recall"]["overall"],  # type:ignore
        "f1": best_val_metrics["f1"]["overall"],  # type:ignore
    }
    return model, {key: value for key, value in sorted(model_evaluation.items())}


def evaluate(running_loss, y_pred, y_true):
    evaluation = {}
    evaluation["loss"] = Model_Metric(overall=running_loss / len(y_true), by_category=None)
    evaluation["acc"] = Model_Metric(
        overall=np.sum(y_pred == y_true) / len(y_true), by_category=None
    )
    evaluation["precision"] = Model_Metric(
        by_category=precision_score(
            y_true, y_pred, average=None, zero_division=0  # type:ignore
        ),
        overall=precision_score(
            y_true, y_pred, average="weighted", zero_division=0  # type:ignore
        ),
    )
    evaluation["recall"] = Model_Metric(
        by_category=recall_score(y_true, y_pred, average=None, zero_division=0),  # type:ignore
        overall=recall_score(
            y_true, y_pred, average="weighted", zero_division=0  # type:ignore
        ),
    )
    evaluation["f1"] = Model_Metric(
        by_category=f1_score(y_true, y_pred, average=None, zero_division=0),  # type:ignore
        overall=f1_score(
            y_true, y_pred, average="weighted", zero_division=0  # type:ignore
        ),
    )
    return evaluation
