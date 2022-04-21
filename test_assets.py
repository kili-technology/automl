from kili.client import Kili


from utils.helpers import get_assets

project_id = "cl0wihlop3rwc0mtj9np28ti2"
kili = Kili()


class TestAssets:
    def test_get_assets(self):
        assets = get_assets(
            kili=kili,
            project_id=project_id,
            label_types=["DEFAULT", "REVIEW"],
            max_assets=20,
            labeling_statuses=["LABELED", "UNLABELED"],
        )
        assert len(assets) == 20

    def test_get_assets_with_labeling_statuses(self):
        assets = get_assets(
            kili=kili,
            project_id=project_id,
            label_types=["DEFAULT", "REVIEW"],
            max_assets=20,
            labeling_statuses=["LABELED"],
        )
        assert len(assets) == 20

        assets = get_assets(
            kili=kili,
            project_id=project_id,
            label_types=["DEFAULT", "REVIEW"],
            max_assets=20,
            labeling_statuses=["UNLABELED"],
        )
        assert len(assets) == 20
