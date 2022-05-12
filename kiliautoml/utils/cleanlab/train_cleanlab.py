import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets

from kiliautoml.utils.path import ModelDirT


def combine_folds(data_dir, model_dir: ModelDirT, verbose=0, num_classes=10, nb_folds=4, seed=42):
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
