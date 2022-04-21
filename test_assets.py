from kili.client import Kili
from utils.helpers import get_assets

project_id = "cl0wihlop3rwc0mtj9np28ti2"
kili = Kili(api_key="2fdd00aa-4825-44cb-b9a7-2a10445cfc18")

test_mock = True


class TestAssets:
    def test_get_assets(self):
        assets = get_assets(
            kili=kili,
            project_id=project_id,
            label_type_in=["DEFAULT", "REVIEW"],
            max_assets=20,
            labeling_statuses=["LABELED", "UNLABELED"],
            test_mock=test_mock,
        )
        assert len(assets) == 20

    def test_get_assets_with_labeling_statuses(self):
        assets = get_assets(
            kili=kili,
            project_id=project_id,
            label_type_in=["DEFAULT", "REVIEW"],
            max_assets=20,
            labeling_statuses=["LABELED"],
            test_mock=test_mock,
        )
        assert len(assets) == 20
