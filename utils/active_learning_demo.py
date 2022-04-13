import os
from typing import List, Any
import pandas as pd
from utils.constants import HOME
import numpy as np

from utils.helpers_functools import kili_print


def get_active_learning_recap_path(project_id):
    return os.path.join(HOME, project_id, "recap_active_learning_csv")


def select_assets_training_active_learning_cycle(project_id, assets):
    """Selects assets for training and active learning cycle."""
    active_learning_recap_path = get_active_learning_recap_path(project_id)
    if os.path.exists(active_learning_recap_path):
        # if the csv exists, we load it and remove the external_ids that have already been labeled
        df = pd.read_csv(active_learning_recap_path)
    else:
        # if the csv does not exist, we create it and add the external_ids to the csv
        external_ids = [a["externalId"] for a in assets]
        df = pd.DataFrame({"external_id": external_ids})
        random_index = np.random.choice(len(external_ids), 100, replace=False)
        df.loc[random_index, "labeled_during_cycle"] = 0  # type: ignore
        df.to_csv(active_learning_recap_path, index=False)

    current_cycle = max(df["labeled_during_cycle"])
    kili_print(f"Current active learning cycle: {current_cycle}")
    external_ids_train = df[df["labeled_during_cycle"] <= current_cycle]["external_id"].to_list()
    kili_print(
        f"Number of asset having been labeled during the past cycle: {len(external_ids_train)}"
    )
    assets = [a for a in assets if a["externalId"] in external_ids_train]
    return assets


def update_recap_active_learning(assets: List[Any], project_id: str, priorities: List[int]):
    """Updates the recap_active_learning_csv file with the new priorities."""
    assert len(assets) == len(priorities)

    active_learning_recap_path = get_active_learning_recap_path(project_id)

    # Get the top most prioritized assets
    index_top_priority = np.argsort(priorities)[::-1]
    external_ids = [assets[i]["externalId"] for i in index_top_priority]
    df = pd.read_csv(active_learning_recap_path)
    assert len(df) >= len(assets)

    # assert no externalIds duplicate
    assert len(df["external_id"].unique()) == len(df)
    assert len(set(external_ids)) == len(external_ids)

    current_cycle = max(df["labeled_during_cycle"]) + 1

    counter_new_added_assets = 0
    for index in index_top_priority:
        mask = df["external_id"] == external_ids[index]
        row = df.loc[mask]

        # either this asset has already been labeled
        asset_cycle = row["labeled_during_cycle"].values[0]
        if not asset_cycle and counter_new_added_assets < 100:
            # or it is the first time we label this asset
            counter_new_added_assets += 1
            df.loc[mask, "labeled_during_cycle"] = current_cycle

        df.loc[mask, f"priority_cycle_{current_cycle}"] = priorities[index]

    df.to_csv(active_learning_recap_path, index=False)
