# %%
from utils.active_learning_demo import (
    N_ASSETS_PER_CYCLE,
    get_active_learning_recap_path,
    get_assets_training_active_learning_cycle,
    update_active_learning_recap,
)
from utils.helpers import get_assets, kili_memoizer, parse_label_types
from kili.client import Kili
import pandas as pd
import os

api_key = os.environ.get("KILI_API_KEY")
project_id = "cl0wihlop3rwc0mtj9np28ti2"


@kili_memoizer
def get_assets_test():
    kili = Kili(api_key=api_key)
    assets = get_assets(
        kili=kili,
        project_id=project_id,
        active_learning_demo=False,
        label_types=parse_label_types(label_types=None),
        labeling_statuses=["LABELED"],
    )
    return assets


def check_recap_active_learning(cycle: int):
    """After each prioritization, we check if the recap csv is updated correctly."""
    df = pd.read_csv(get_active_learning_recap_path(project_id))
    print(df.dropna().shape[0], "assets scheduled in recap csv")
    assert df.dropna().shape[0] == (cycle + 1) * N_ASSETS_PER_CYCLE
    assert int(df["labeled_during_cycle"].dropna().max()) == cycle


def test_select_assets_training_active_learning_cycle():

    # Initialization
    all_assets = get_assets_test()
    priorities = list(range(len(all_assets)))

    # %% Cycle - 0
    # Training and Initialization
    _ = get_assets_training_active_learning_cycle(
        project_id=project_id, all_assets=all_assets, cycle_0=True
    )
    check_recap_active_learning(cycle=0)

    # %%
    # Prioritization
    update_active_learning_recap(all_assets, project_id=project_id, priorities=priorities)
    check_recap_active_learning(cycle=1)

    # %% Cycle - 1
    # Training and Initialization
    _ = get_assets_training_active_learning_cycle(project_id=project_id, all_assets=all_assets)
    check_recap_active_learning(cycle=1)

    # Prioritization
    update_active_learning_recap(all_assets, project_id=project_id, priorities=priorities)
    check_recap_active_learning(cycle=2)
