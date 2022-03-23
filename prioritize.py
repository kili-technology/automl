import os
from typing import List

import click
from PIL.Image import Image as PILImage
import torch
import numpy as np
from img2vec_pytorch import Img2Vec
from kili.client import Kili


from utils.constants import (
    InputType,
)
from utils.helpers import (
    clear_automl_cache,
    download_project_images,
    get_assets,
    get_project,
    kili_print,
    parse_label_types,
)


class Prioritizer:
    def __init__(self, embeddings: np.ndarray):
        assert len(embeddings) > 0 and len(embeddings[0]) > 0
        self.embeddings = embeddings

    def get_priorities_diversity_sampling(self) -> List[int]:
        """Implement diversity sampling

        Cluster-based sampling

        returns a list of priority.
        """
        import numpy as np
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans  #

        embeddings = self.embeddings

        pipe = Pipeline(
            [("pca", PCA(n_components=10)), ("kmeans", KMeans(n_clusters=5))]
        )

        X_clusters = pipe.fit_transform(embeddings)[:, 0]

        index_clusters = {
            i: np.where(X_clusters == i)[0] for i in np.unique(X_clusters)
        }
        index_clusters_permuted = {
            i: np.random.permutation(index_clusters[i]).tolist() for i in index_clusters
        }

        sampling = []
        for _ in range(len(embeddings)):
            cluster_id = np.random.choice(list(index_clusters_permuted.keys()))
            index = index_clusters_permuted[cluster_id].pop(0)

            if len(index_clusters_permuted[cluster_id]) == 0:
                del index_clusters_permuted[cluster_id]

            sampling.append(index)

        priorities = np.argsort(np.array(sampling))

        return priorities.tolist()

    def get_random_sampling_priorities(self):
        """Implement random sampling

        returns a list of priority.
        """
        assert len(self.embeddings) > 0
        return np.random.permutation(len(self.embeddings)).tolist()

    @staticmethod
    def get_combine_priorities(priorities_a: List[int], priorities_b: List[int], proba_a: float = 0.5):
        """Combine two priority lists
        
        Sample from the first list with proba coef_a
        """
        assert len(priorities_a) == len(priorities_b)
        priorities = []
        for i in range(len(priorities_a)):
            priorities.append(priorities_a[i] + priorities_b[i])
        return priorities

    def get_priorities(self, diversity_sampling: float) -> List[int]:
        """diversity_sampling is a float between 0 and 1"""
        assert 0 <= diversity_sampling <= 1

        random_sampling = 1 - diversity_sampling
        kili_print(
            f"Sampling Mix of {diversity_sampling*100}% of  Diversity Sampling "
            f"and {random_sampling*100}% of Random Sampling"
        )

        diversity_sampling_priorities = self.get_priorities_diversity_sampling()
        random_sampling_priorities = self.get_random_sampling_priorities()

        priorities = self.get_combine_priorities(
            diversity_sampling_priorities, random_sampling_priorities
        )

        return priorities


def embeddings_images(
    images: List[PILImage],
) -> np.ndarray:
    """Get the embeddings of the images using a generic model trained on ImageNet."""
    img2vec = Img2Vec(cuda=torch.cuda.is_available())
    vectors = img2vec.get_vec(images)
    return vectors


def embeddings_ner(
    texts: List[str],
):
    raise NotImplementedError


def embedding_text(
    texts: List[str],
):
    raise NotImplementedError


@click.command()
@click.option("--api-key", default=os.environ.get("KILI_API_KEY"), help="Kili API Key")
@click.option(
    "--project-id", default=os.environ.get("PROJECT_ID"), help="Kili project ID"
)
@click.option(
    "--label-types",
    default=None,
    help="Comma separated list Kili specific label types to select (among DEFAULT, REVIEW, PREDICTION)",
)
@click.option(
    "--max-assets",
    default=None,
    type=int,
    help="Maximum number of assets to consider",
)
@click.option(
    "--diversity-sampling",
    default=0.4,
    type=float,
    help="Diversity sampling proportion",
)
@click.option(
    "--clear-dataset-cache",
    default=False,
    is_flag=True,
    help="Tells if the dataset cache must be cleared",
)
def main(
    api_key: str,
    project_id: str,
    label_types: str,
    max_assets: int,
    diversity_sampling: float,
    clear_dataset_cache: bool,
):
    """
    Prioritize assets in a Kili project.

    The goal is to find a collection of assets that are most relevant to prioritize.
    We embedded the assets using a generic model, and then use a strategy that is a mixture of
    diversity sampling, uncertainty sampling, and random sampling to sorts the assets.
    """
    kili = Kili(api_key=api_key)
    input_type, jobs = get_project(kili, project_id)
    kili_print("Input type: ", input_type)
    kili_print("jobs: ", jobs)

    if clear_dataset_cache:
        clear_automl_cache()

    unlabeled_assets = get_assets(
        kili,
        project_id,
        parse_label_types(label_types),
        max_assets,
        get_labeled=False,
    )

    if input_type == InputType.Image:

        downloaded_images = download_project_images(api_key, unlabeled_assets)

        pil_images = [image.image for image in downloaded_images]

        embeddings = embeddings_images(pil_images)
        kili_print("Embeddings successfully computed with shape ", embeddings.shape)

    else:
        raise NotImplementedError

    asset_ids = [asset["id"] for asset in unlabeled_assets]
    prioritizer = Prioritizer(embeddings)
    priorities = prioritizer.get_priorities(diversity_sampling=diversity_sampling)
    print(priorities)
    kili.update_properties_in_assets(asset_ids=asset_ids, priorities=priorities)


if __name__ == "__main__":
    main()
