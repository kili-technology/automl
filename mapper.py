import os
from typing import List, Optional

import click
from kili.client import Kili

from kiliautoml.utils.helpers import get_assets, get_project, kili_print
from kiliautoml.utils.mapper.create import MapperClassification
from kiliautoml.utils.type import AssetStatusT, LabelMergeStrategyT


@click.command()
@click.option(
    "--api-endpoint",
    default="https://cloud.kili-technology.com/api/label/v2/graphql",
    help="Kili Endpoint",
)
@click.option("--api-key", default=os.environ.get("KILI_API_KEY", ""), help="Kili API Key")
@click.option("--project-id", default=None, required=True, help="Kili project ID")
@click.option(
    "--target-job",
    default=None,
    multiple=True,
    help=(
        "Add a specific target job on which to focus on "
        "(multiple can be passed if --target-job <job_name> is repeated) "
        "Example: python mapper.py --target-job CLASSIFICATION0 --target-job CLASSIFICATION1"
    ),
)
@click.option(
    "--asset-status-in",
    default=None,
    callback=lambda _, __, x: x.upper().split(",") if x else None,
    help=(
        "Comma separated (without space) list of Kili asset status to select "
        "among: 'TODO', 'ONGOING', 'LABELED', 'TO_REVIEW', 'REVIEWED'"
        "Example: python mapper.py --asset-status-in TO_REVIEW,REVIEWED "
    ),
)
@click.option(
    "--label-merge-strategy",
    default="last",
    help=(
        "Strategy to select the right label when more than one are available"
        "for one asset. AutoML always select the best type of label ('Review' then "
        "'Default'). When there are several labels for the highest priority label type, "
        "the user can specify if the last label is taken or the first one"
    ),
)
@click.option(
    "--max-assets",
    default=None,
    type=int,
    help="Maximum number of assets to consider",
)
@click.option("--assets-repository", default=None, help="Asset repository (eg. /content/assets/)")
@click.option(
    "--predictions-path", default=None, help="csv file with predictions (first column = asset_id)"
)
@click.option(
    "--cv-folds",
    default=4,
    type=int,
    show_default=True,
    help="Number of CV folds to use if all data are labeled",
)
@click.option(
    "--focus-class",
    default=None,
    callback=lambda _, __, x: x.split(",") if x else None,
    help="Only display selected class in Mapper graph",
)
def main(
    api_endpoint: str,
    api_key: str,
    project_id: str,
    #   clear_dataset_cache: bool,
    target_job: List[str],
    #    model_name: ModelNameT,
    #    model_repository: ModelRepositoryT,
    asset_status_in: Optional[List[AssetStatusT]],
    label_merge_strategy: LabelMergeStrategyT,
    max_assets: int,
    assets_repository: str,
    predictions_path: Optional[str],
    cv_folds: int,
    focus_class: Optional[List[str]],
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
        ml_task = job.get("mlTask")
        if content_input == "radio" and ml_task == "CLASSIFICATION":
            # Get assets
            assets = get_assets(
                kili,
                project_id,
                status_in=asset_status_in,
                max_assets=max_assets,
            )

            mapper_image_classification = MapperClassification(
                api_key=api_key,
                input_type=input_type,
                assets=assets,
                job=job,
                job_name=job_name,
                assets_repository=assets_repository,
                asset_status_in=asset_status_in,
                label_merge_strategy=label_merge_strategy,
                predictions_path=predictions_path,
                focus_class=focus_class,
            )

            _ = mapper_image_classification.create_mapper(cv_folds)

        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
