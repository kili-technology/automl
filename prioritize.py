import os
from typing import List, Optional

import click
import numpy as np
import torch
from img2vec_pytorch import Img2Vec
from kili.client import Kili
from more_itertools import chunked
from numpy.testing import assert_almost_equal
from PIL.Image import Image as PILImage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from kiliautoml.utils.constants import InputType, MLTaskT, ModelFrameworkT, ToolT
from kiliautoml.utils.download_assets import download_project_images
from kiliautoml.utils.helpers import (
    get_assets,
    get_project,
    kili_print,
    parse_label_types,
)
from kiliautoml.utils.memoization import clear_automl_cache
from predict import predict_one_job

# Priorities
Priorities = List[float]

# Normalized priorities (between [0, length of queue])
PrioritiesNormalized = List[int]


class PriorityQueue:
    """List of asset_index beginning with highest priority

    Compared to the queue in the standard library, this queue adds the method remove.
    """

    def __init__(self, priorities: Priorities):
        self.queue = list(np.argsort(priorities)[::-1].tolist())
        # example queue[0] is the index of the highest priority asset
        self.check_queue_validity()

    def pop(self):
        """Get the asset_index of the asset with highest priority"""
        return self.queue.pop(0)

    def remove(self, asset_index: int):
        """Remove asset_index from the queue"""
        self.queue.remove(asset_index)
        self.check_queue_validity()

    def append(self, asset_index: int):
        """Append asset_index after the queue."""
        self.queue.append(asset_index)
        self.check_queue_validity()

    def is_empty(self):
        return len(self.queue) == 0

    def to_prioritized_list(self) -> PrioritiesNormalized:
        priority = [0] * len(self.queue)
        for i, asset_index in enumerate(self.queue):
            # i = 0 mean top-0 priority
            priority[asset_index] = len(self.queue) - 1 - i
        return priority

    def check_queue_validity(self):
        # no duplicates
        assert len(self.queue) == len(set(self.queue))

        # only integer values
        assert all([isinstance(asset_index, int) for asset_index in self.queue])

    def __repr__(self):
        return f"Queue({self.queue}) (first element is most prioritized)"


def pop_queues(queue_a: PriorityQueue, queue_b: PriorityQueue, queue: PriorityQueue):
    """Pop the highest priority asset_index from queue_a append to queue.

    remove the asset_index from queue_b.
    """
    asset_index = queue_a.pop()
    queue.append(asset_index)
    queue_b.remove(asset_index)


def normalize_priorities(
    priorities: Priorities,
) -> PrioritiesNormalized:
    """Normalize the priorities between [Ø, length of queue]"""
    return PriorityQueue(priorities).to_prioritized_list()


class Prioritizer:
    def __init__(self, embeddings: np.ndarray, predictions_probability: List[float] = []):
        """Initialize Prioritizer class.

        Args:
            embeddings: matrix of embeddings
            predictions_probability: list of probability

        Raises:
        ConnectionError: If no available port is found.
        """
        assert len(embeddings) > 0 and len(embeddings[0]) > 0
        self.embeddings = embeddings
        self.predictions_probability = predictions_probability

        if predictions_probability:
            assert len(embeddings) == len(predictions_probability)

    def _get_priorities_diversity_sampling(self) -> PrioritiesNormalized:
        """Implement diversity sampling.

        We cluster the embeddings and then we create a priority list for each cluster.
        Then we combine the priority lists.

        Returns:
            List[int]: list of priorities.
        """
        embeddings = self.embeddings

        pipe = Pipeline([("pca", PCA(n_components=10)), ("kmeans", KMeans(n_clusters=5))])

        X_clusters = pipe.fit_transform(embeddings)[:, 0]

        index_clusters = {i: np.where(X_clusters == i)[0] for i in np.unique(X_clusters)}
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

    def _get_priorities_uncertainty_sampling(self) -> PrioritiesNormalized:
        """Implement uncertainty sampling with least Confidence strategy.

        Least Confidence: difference between the most confident prediction and 100% confidence.

        Returns:
            List[int]: list of priorities.
        """
        predictions_probability = self.predictions_probability
        if not predictions_probability:
            ValueError("No predictions_probability available.")
        prio = list(-np.array(predictions_probability))

        # We normalize the priorites to be between 0 and len(list)
        queue = PriorityQueue(prio)
        return queue.to_prioritized_list()

    def _get_random_sampling_priorities(self) -> PrioritiesNormalized:
        """Implement random sampling"""
        assert len(self.embeddings) > 0
        return np.random.permutation(len(self.embeddings)).tolist()

    @staticmethod
    def combine_priorities(
        priorities_a: Priorities, priorities_b: Priorities, proba_a: float = 0.5
    ) -> PrioritiesNormalized:
        """Combine two priority lists.

        Args:
            priorities_a (List[int]): first priority list
            priorities_b (List[int]): second priority list
            proba_a (float, optional): probability of taking the first priority list.

        Warning:
            This function returns a normalized priority list.

        Returns:
            List[int]: combined priority list
        """
        assert len(priorities_a) == len(priorities_b)
        assert 0 <= proba_a <= 1

        queue_a = PriorityQueue(priorities_a)
        queue_b = PriorityQueue(priorities_b)

        queue_combined = PriorityQueue([])
        for _ in priorities_a:
            if np.random.random() < proba_a:
                pop_queues(queue_a, queue_b, queue_combined)
            else:
                pop_queues(queue_b, queue_a, queue_combined)

        return queue_combined.to_prioritized_list()

    @staticmethod
    def combine_multiple_priorities(
        priorities: List[Priorities], probas: List[float]
    ) -> PrioritiesNormalized:
        assert_almost_equal(np.sum(probas), 1)

        n_assets = len(priorities[0])
        assert max([len(x) for x in priorities]) == n_assets
        assert min([len(x) for x in priorities]) == n_assets

        assert len(probas) == len(priorities)

        if len(probas) == 1:
            return normalize_priorities(priorities[0])
        elif len(probas) == 2:
            return Prioritizer.combine_priorities(priorities[0], priorities[1], probas[0])
        else:
            argsort = np.argsort(np.array(probas))

            priorities_sorted = [priorities[i] for i in argsort]
            probas_sorted = [probas[i] for i in argsort]

            if probas_sorted[0] < 0.0001:
                # The first priority queue should be ignored
                return Prioritizer.combine_multiple_priorities(
                    priorities_sorted[1:], probas_sorted[1:]
                )
            else:
                # All priority queues should be considered
                other_probas = np.array(probas_sorted[:2])
                other_priorities = Prioritizer.combine_multiple_priorities(
                    priorities_sorted[:2], other_probas / np.sum(other_probas)
                )
                return Prioritizer.combine_priorities(
                    priorities_sorted[0],
                    other_priorities,  # type: ignore
                    probas_sorted[0],
                )

    def get_priorities(
        self, diversity_sampling: float, uncertainty_sampling: float
    ) -> PrioritiesNormalized:
        """diversity_sampling is a float between 0 and 1"""
        assert 0 <= diversity_sampling <= 1
        assert 0 <= uncertainty_sampling <= 1
        assert 0 <= uncertainty_sampling + diversity_sampling <= 1

        random_sampling = 1 - diversity_sampling - uncertainty_sampling
        kili_print(
            f"Sampling Mix of {diversity_sampling*100}% of  Diversity Sampling "
            f"and {uncertainty_sampling*100}% of Uncertainty Sampling "
            f"and {random_sampling*100}% of Random Sampling."
        )

        diversity_sampling_priorities = self._get_priorities_diversity_sampling()
        uncertainty_sampling_priorities = self._get_priorities_uncertainty_sampling()
        random_sampling_priorities = self._get_random_sampling_priorities()

        priorities = [
            diversity_sampling_priorities,
            uncertainty_sampling_priorities,
            random_sampling_priorities,
        ]

        combined_priorities = self.combine_multiple_priorities(
            priorities,  # type: ignore
            [
                diversity_sampling,
                uncertainty_sampling,
                random_sampling,
            ],
        )

        return combined_priorities


def embeddings_images(images: List[PILImage], batch_size=4) -> np.ndarray:
    """Get the embeddings of the images using a generic model trained on ImageNet."""
    color_images = [im.convert("RGB") for im in images]
    img2vec = Img2Vec(cuda=torch.cuda.is_available(), model="efficientnet_b7")
    vecs = []
    for imgs in tqdm(list(chunked(color_images, batch_size))):
        _ = np.array(img2vec.get_vec(imgs))
        vecs.append(_)
    return np.concatenate(vecs, axis=0)


def embeddings_ner(
    texts: List[str],
):
    raise NotImplementedError


def embedding_text(
    texts: List[str],
):
    raise NotImplementedError


@click.command()
@click.option(
    "--api-endpoint",
    default="https://cloud.kili-technology.com/api/label/v2/graphql",
    help="Kili Endpoint",
)
@click.option("--api-key", default=os.environ.get("KILI_API_KEY"), help="Kili API Key")
@click.option("--project-id", required=True, help="Kili project ID")
@click.option(
    "--label-types",
    default=None,
    help=(
        "Comma separated list Kili specific label types to select "
        "(among DEFAULT, REVIEW, PREDICTION)"
    ),
)
@click.option(
    "--max-assets",
    default=None,
    type=int,
    help="Maximum number of assets to consider",
)
@click.option(
    "--diversity-sampling",
    default=0.3,
    type=float,
    help="Diversity sampling proportion",
)
@click.option(
    "--uncertainty-sampling",
    default=0.4,
    type=float,
    help="Uncertainty sampling proportion",
)
@click.option(
    "--dry-run",
    default=None,
    is_flag=True,
    help="Runs the predictions but do not save them into the Kili project",
)
@click.option(
    "--from-model",
    default=None,
    help="Runs the predictions using a specified model path",
)
@click.option(
    "--verbose",
    default=0,
    help="Verbose level",
)
@click.option(
    "--clear-dataset-cache",
    default=False,
    is_flag=True,
    help="Tells if the dataset cache must be cleared",
)
@click.option(
    "--from-project",
    default=None,
    type=str,
    help=(
        "Use a model trained of a different project to predict on project_id."
        "This is usefull if you do not want to pollute the original project with "
        "experimental predictions."
        "This argument is ignored if --from-model is used."
    ),
)
def main(
    api_endpoint: str,
    api_key: str,
    project_id: str,
    label_types: str,
    max_assets: Optional[int],
    diversity_sampling: float,
    uncertainty_sampling: float,
    dry_run: bool,
    from_model: ModelFrameworkT,
    verbose: bool,
    clear_dataset_cache: bool,
    from_project: Optional[str],
    model_name: Optional[str],
    model_repository: Optional[str],
):
    """
    Prioritize assets in a Kili project.

    The goal is to find a collection of assets that are most relevant to prioritize.
    We embedded the assets using a generic model, and then use a strategy that is a mixture of
    diversity sampling, uncertainty sampling, and random sampling to sorts the assets.
    """
    if uncertainty_sampling + diversity_sampling > 1:
        raise ValueError("diversity_sampling + diversity_sampling should be less than 1.")

    if max_assets and max_assets < 10:
        raise ValueError("max_assets should be greater than 10")

    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, _ = get_project(kili, project_id)
    kili_print("Input type: ", input_type)
    kili_print("jobs: ", jobs)

    jobs_item = list(jobs.items())
    if len(jobs_item) > 1:
        raise NotImplementedError

    job_name, job = jobs_item[0]
    content_input = job.get("content", {}).get("input")
    ml_task: MLTaskT = job.get("mlTask")
    tools: ToolT = job.get("tools")

    if clear_dataset_cache:
        clear_automl_cache(
            command="prioritize", project_id=project_id, job_name=job_name, model_repository=None
        )

    unlabeled_assets = get_assets(
        kili,
        project_id,
        parse_label_types(label_types),
        max_assets,
        labeling_statuses=["UNLABELED"],
    )

    # TODO: useless if uncertainty_sampling == 0
    job_predictions = predict_one_job(
        api_key=api_key,
        api_endpoint=api_endpoint,
        project_id=project_id,
        from_model=from_model,
        verbose=verbose,
        input_type=input_type,
        assets=unlabeled_assets,
        job_name=job_name,
        content_input=content_input,
        model_repository=model_repository,
        model_name=model_name,
        ml_task=ml_task,
        tools=tools,
        prioritization=True,
        from_project=from_project,
    )

    if input_type == InputType.Image:
        downloaded_images = download_project_images(
            api_key, unlabeled_assets, project_id, output_folder=None
        )
        pil_images = [image.image for image in downloaded_images]
        embeddings = embeddings_images(pil_images)
        kili_print("Embeddings successfully computed with shape ", embeddings.shape)
    else:
        raise NotImplementedError

    predictions_probability = job_predictions.predictions_probability
    print("Predictions probability shape: ", predictions_probability)
    asset_ids = [asset["id"] for asset in unlabeled_assets]
    prioritizer = Prioritizer(embeddings, predictions_probability=predictions_probability)
    priorities = prioritizer.get_priorities(
        diversity_sampling=diversity_sampling, uncertainty_sampling=uncertainty_sampling
    )
    if not dry_run:
        kili.update_properties_in_assets(asset_ids=asset_ids, priorities=priorities)


if __name__ == "__main__":
    main()
