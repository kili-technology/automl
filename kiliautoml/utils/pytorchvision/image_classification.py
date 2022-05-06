from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_Data
from torchvision import models

from kiliautoml.utils.constants import ModelName, ModelNameT, ModelRepositoryT
from kiliautoml.utils.helpers import set_default
from kiliautoml.utils.path import ModelPathT
from kiliautoml.utils.pytorchvision.trainer import train_model_pytorch


def set_model_name_image_classification(model_name) -> ModelNameT:
    model_name = set_default(  # type:ignore
        model_name,
        ModelName.EfficientNetB0,
        "model_name",
        [ModelName.EfficientNetB0, ModelName.Resnet50],
    )

    return model_name


def set_model_repository_image_classification(model_repository) -> ModelRepositoryT:
    model_repository = set_default(
        model_repository,
        "torchvision",
        "model_repository",
        ["torchvision"],
    )

    return model_repository


def get_trained_model_image_classif(
    epochs: int,
    model_name: ModelNameT,
    verbose: int,
    class_names: list,
    image_datasets: dict,
    save_model_path: Optional[ModelPathT] = None,
):
    dataloaders = {
        x: torch_Data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=1)
        for x in ["train", "val"]
    }

    model = initialize_model_img_class(model_name, class_names)

    model, loss = train_model_pytorch(
        model=model,
        dataloaders=dataloaders,
        verbose=verbose,
        epochs=epochs,
    )

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)

    return model, loss


def initialize_model_img_class(model_name: ModelNameT, class_names):
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
    return model


def predict_probabilities(loader: torch_Data.DataLoader, model, verbose=0) -> List[float]:
    """
    Method to compute the probabilities for all classes for the assets in the holdout set
    """
    # Switch to evaluate mode.
    model.eval()
    n_total = len(loader.dataset.imgs) / float(loader.batch_size)  # type:ignore
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
