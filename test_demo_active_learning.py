from utils.active_learning_demo import (
    get_active_learning_recap_path,
    select_assets_training_active_learning_cycle,
)
from utils.helpers import get_assets, parse_label_types
from kili.client import Kili
import pandas as pd


def get_assets_test():
    kili = Kili(api_key="93d8d3fa-ed06-410c-b295-9a9310e88fb0")
    assets = get_assets(
        kili=kili,
        project_id="écl0wihlop3rwc0mtj9np28ti2",
        label_types=parse_label_types(label_types=None),
        max_assets=None,
        labeling_statuses=["LABELED"],
    )
    return assets


assets = get_assets_test()
assets = select_assets_training_active_learning_cycle(project_id="test_project", assets=assets)

path = get_active_learning_recap_path(project_id="test_project")
pd.read_csv(path)
