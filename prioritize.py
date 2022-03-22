from email.policy import default
import os
from  pprint import pprint

import click
from PIL.Image import open, Image
import numpy as np

from kili.client import Kili
from utils.constants import (
    InputType,
)
from utils.helpers import (
    get_assets,
    get_project,
    kili_print,
    parse_label_types,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def embeddings_images(
    images : list[Image],
) -> np.ndarray:
    """Get the embeddings of the images using a generic model trained on ImageNet."""
    from img2vec_pytorch import Img2Vec
    img2vec = Img2Vec(cuda=True)
    vectors = img2vec.get_vec(images)
    return vectors


def embeddings_ner(
    texts : list[str],
):
    raise NotImplementedError


def embedding_text(
    texts : list[str],
):
    raise NotImplementedError

@click.command()
@click.option("--api-key", default=os.environ["KILI_API_KEY"], help="Kili API Key")
@click.option("--project-id", default=os.environ["PROJECT_ID"], help="Kili project ID")
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
   "--diversity-sampling", default=0.4, type=float, help="Diversity sampling proportion"
)
def main(
    api_key: str,
    project_id: str,
    label_types: str,
    max_assets: int,
    diversity_sampling: float
):
    """ """
    kili = Kili(api_key=api_key)
    input_type, jobs = get_project(kili, project_id)
    pprint("Input type: ",input_type)
    pprint("jobs:", jobs)

    assets = get_assets(kili, project_id, parse_label_types(label_types), max_assets)

    if input_type == InputType.Image:
        print(assets)
        im = open("images/predict.png")
        images = [im]
        embeddings = embeddings_images(images)
        kili_print("Embeddings successfully computed with shape ", embeddings.shape)

        random_sampling = 1 - diversity_sampling
        kili_print(
            f"Sampling Mix of {diversity_sampling*100}% of  Diversity Sampling "
            f"and {random_sampling*100}% of Random Sampling"
        )

    else:
        raise NotImplementedError

        


if __name__ == "__main__":
    main()
