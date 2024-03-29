from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_Data
import wandb
from torch.utils.data import Dataset
from torchvision import models, transforms

from kiliautoml.utils.download_assets import DownloadedImage
from kiliautoml.utils.logging import logger
from kiliautoml.utils.path import ModelPathT
from kiliautoml.utils.pytorchvision.trainer import train_model_pytorch
from kiliautoml.utils.type import CategoryIdT, ModelNameT

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


class ClassificationTrainDataset(Dataset):  # type: ignore
    def __init__(
        self,
        images: List[DownloadedImage],
        labels: List[CategoryIdT],
        class_name_to_idx: Dict[CategoryIdT, int],
        transform=None,
    ):
        """
        Args:
            Images (list of DownloadedImages)
            Labels (list of string)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.labels = labels
        self.class_name_to_idx = class_name_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx].get_image()
        image = image.convert("RGB")
        label_idx = self.class_name_to_idx[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label_idx


class ClassificationPredictDataset(Dataset):  # type: ignore
    def __init__(self, images: List[DownloadedImage], transform=None):
        """
        Args:
            Images (list of DownloadedImages)
            Labels (list of string)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx].get_image()
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def get_trained_model_image_classif(
    *,
    epochs: int,
    model_name: ModelNameT,
    batch_size: int,
    category_ids: List[CategoryIdT],
    image_datasets: dict,  # type: ignore
    save_model_path: Optional[ModelPathT] = None,
    disable_wandb: bool = False,
):
    dataloaders = {
        x: torch_Data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1
        )
        for x in ["train", "val"]
    }

    model = initialize_model_img_class(model_name, category_ids)
    if not disable_wandb:
        wandb.watch(model)  # type: ignore[attr-defined]

    model, model_evaluation = train_model_pytorch(
        model=model,
        dataloaders=dataloaders,
        epochs=epochs,
        class_names=category_ids,
        disable_wandb=disable_wandb,
    )

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)

        if not disable_wandb:
            artifact = wandb.Artifact("model", type="model")  # type: ignore[attr-defined]
            artifact.add_file(save_model_path)
            wandb.log_artifact(artifact)  # type: ignore[attr-defined]
            wandb.join()  # type: ignore[attr-defined]

    return model, model_evaluation


def initialize_model_img_class(model_name: ModelNameT, class_names):
    logger.info("Initialization of the model with N={} classes".format(len(class_names)))
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))  # type:ignore
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model


def predict_probabilities(
    loader: torch_Data.DataLoader,  # type: ignore
    model,
) -> List[float]:
    """
    Method to compute the probabilities for all classes for the assets in the holdout set
    """
    # Switch to evaluate mode.
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_total = len(loader.dataset) / float(loader.batch_size)  # type:ignore
    outputs = []
    logger.debug("Computing probabilities for this fold with device: {}".format(device))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print("\rComplete: {:.1%}".format(i / n_total), end="")
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)

            # compute output
            outputs.append(model(input))
        logger.debug("")

    # Prepare outputs as a single matrix
    probs = list(
        np.concatenate(
            [torch.nn.functional.softmax(z, dim=1).cpu().numpy() for z in outputs]  # type:ignore
        )
    )
    return probs
