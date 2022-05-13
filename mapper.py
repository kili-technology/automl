import os
from typing import List

import click
from kili.client import Kili

from kiliautoml.utils.constants import (  # ModelFrameworkT,; ModelNameT,; ModelRepositoryT,
    ContentInput,
    InputType,
    MLTask,
    MLTaskT,
)
from kiliautoml.utils.helpers import (
    get_assets,
    get_project,
    kili_print,
    parse_label_types,
)
from kiliautoml.utils.mapper.create import MapperImageClassification
from kiliautoml.utils.type import LabelTypeT


@click.option(
    "--api-endpoint",
    default="https://cloud.kili-technology.com/api/label/v2/graphql",
    help="Kili Endpoint",
)
@click.option("--api-key", default=os.environ.get("KILI_API_KEY", ""), help="Kili API Key")
@click.option(
    "--target-job",
    default=None,
    multiple=True,
    help=(
        "Add a specific target job on which to train on "
        "(multiple can be passed if --target-job <job_name> is repeated) "
        "Example: python train.py --target-job BBOX --target-job CLASSIFICATION"
    ),
)
@click.option(
    "--label-types",
    default=None,
    help=(
        "Comma separated list Kili specific label types to select (among DEFAULT,"
        " REVIEW, PREDICTION)"
    ),
)
@click.option(
    "--max-assets",
    default=None,
    type=int,
    help="Maximum number of assets to consider",
)
@click.option("--assets-repository", default=None, help="Asset repository (eg. /content/assets/)")
@click.option("--project-id", default=None, required=True, help="Kili project ID")
@click.option(
    "--cv-folds",
    default=4,
    type=int,
    show_default=True,
    help="Number of CV folds to use if all data are labeled",
)
def main(
    api_endpoint: str,
    api_key: str,
    project_id: str,
    #   clear_dataset_cache: bool,
    target_job: List[str],
    #    model_name: ModelNameT,
    #    model_repository: ModelRepositoryT,
    label_types: LabelTypeT,
    max_assets: int,
    assets_repository: str,
    cv_folds: int,
):
    """
    Main method for creating mapper
    """

    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, _ = get_project(kili, project_id)

    if max_assets and max_assets < 10:
        raise ValueError("max_assets should be greater than 10")

    for job_name, job in jobs.items():
        if target_job and job_name not in target_job:
            continue

        kili_print(f"Create Mapper for job: {job_name}")

        content_input = job.get("content", {}).get("input")
        ml_task: MLTaskT = job.get("mlTask")
        if (
            content_input == ContentInput.Radio
            and input_type == InputType.Image
            and ml_task == MLTask.Classification
        ):
            # Get assets
            assets = get_assets(
                kili,
                project_id,
                parse_label_types(label_types),
                labeling_statuses=["UNLABELED", "LABELED"],
                max_assets=max_assets,
            )

            mapper_image_classification = MapperImageClassification(
                api_key=api_key,
                project_id=project_id,
                assets=assets,
                job=job,
                job_name=job_name,
                assets_repository=assets_repository,
                label_types=label_types,
            )

            _ = mapper_image_classification.create_mapper(cv_folds)

        else:
            raise NotImplementedError
