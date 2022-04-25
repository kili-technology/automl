"""
select_assets_training_active_learning_cycle
"""
import os
from typing import List, Any
import pandas as pd
from utils.constants import HOME
import numpy as np

from utils.helpers_functools import kili_print


N_ASSETS_PER_CYCLE = 100


def get_active_learning_recap_path(project_id):
    res = os.path.join(HOME, project_id, "recap_active_learning.csv")

    dir_path = os.path.join(HOME, project_id)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    kili_print(f"Active learning recap path: {res}")
    return res


def get_assets_training_active_learning_cycle(
    project_id: str, all_assets: List[Any], cycle_0=False
):
    """Selects assets for training the current active learning cycle."""
    active_learning_recap_path = get_active_learning_recap_path(project_id)
    if os.path.exists(active_learning_recap_path) and not cycle_0:
        # if the csv exists, we load it and remove the external_ids that have already been labeled
        df = pd.read_csv(active_learning_recap_path)
    else:
        # if the csv does not exist, we create it and add the external_ids to the csv
        external_ids = [a["externalId"] for a in all_assets]
        df = pd.DataFrame({"external_id": external_ids})
        random_index = np.random.choice(len(external_ids), N_ASSETS_PER_CYCLE, replace=False)
        df.loc[random_index, "labeled_during_cycle"] = 0  # type: ignore
        df.to_csv(active_learning_recap_path, index=False)

    current_cycle = max(df["labeled_during_cycle"].dropna())
    kili_print(f"Current active learning cycle: {current_cycle}")
    external_ids_train = df[df["labeled_during_cycle"] <= current_cycle]["external_id"].to_list()
    train_assets = [a for a in all_assets if a["externalId"] in external_ids_train]
    kili_print(f"Number of assets selected for training: {len(train_assets)}")
    return train_assets


def save_prioritization(assets: List[Any], project_id: str, priorities: List[int]):
    """Write on the recaping csv the labels which will be labeled during the
    next active learning cycle."""
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

    next_cycle = int(max(df["labeled_during_cycle"].dropna())) + 1

    counter_new_added_assets = 0
    for index in index_top_priority:
        mask = df["external_id"] == external_ids[index]
        row = df.loc[mask]

        asset_cycle = row["labeled_during_cycle"].values[0]
        asset_not_labeled = bool(pd.isnull(asset_cycle))

        # check this asset has already been labeled
        if asset_not_labeled and counter_new_added_assets < N_ASSETS_PER_CYCLE:
            counter_new_added_assets += 1
            df.loc[mask, "labeled_during_cycle"] = next_cycle

        df.loc[mask, f"priority_cycle_{next_cycle}"] = priorities[index]
    print("Active learning cycle scheduled:", set(df.labeled_during_cycle.dropna()))
    df.to_csv(active_learning_recap_path, index=False)
    print("Active learning cycle recap saved in ", active_learning_recap_path)
