import os
from typing import List

import click
from kili.client import Kili

from kiliautoml.models._pytorchvision_image_classification import (
    PyTorchVisionImageClassificationModel,
)
from kiliautoml.utils.constants import (
    MLTaskT,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
)
from kiliautoml.utils.helpers import (
    get_assets,
    get_project,
    kili_print,
    upload_errors_to_kili,
)
from kiliautoml.utils.memoization import clear_automl_cache
from kiliautoml.utils.type import AssetStatusT


@click.command()
@click.option(
    "--api-endpoint",
    default="https://cloud.kili-technology.com/api/label/v2/graphql",
    help="Kili Endpoint",
)
@click.option("--api-key", default=os.environ.get("KILI_API_KEY"), help="Kili API Key")
@click.option(
    "--cv-folds", default=4, type=int, show_default=True, help="Number of CV folds to use"
)
@click.option("--model-framework", default=None, help="Model framework (eg. pytorch, tensorflow)")
@click.option("--model-repository", default=None, help="Model repository (eg. huggingface)")
@click.option(
    "--clear-dataset-cache",
    default=False,
    is_flag=True,
    help="Tells if the dataset cache must be cleared",
)
@click.option(
    "--target-job",
    default=None,
    multiple=True,
    help=(
        "Add a specific target job for which to detect the errors "
        "(multiple can be passed if --target-job <job_name> is repeated)"
        "Example: python label_errors.py --target-job BBOX --target-job CLASSIFICATION"
    ),
)
@click.option(
    "--dry-run",
    default=None,
    is_flag=True,
    help=(
        "Get the labeling errors, save on the hard drive, but do not upload them into the Kili"
        " project"
    ),
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    show_default=True,
    help="Number of epochs to train each CV fold",
)
@click.option(
    "--asset-status-in",
    default=["LABELED", "TO_REVIEW", "REVIEWED"],
    callback=lambda _, __, x: x.upper().split(",") if x else [],
    help=(
        "Comma separated (without space) list of Kili asset status to select "
        "among: 'TODO', 'ONGOING', 'LABELED', 'TO_REVIEW', 'REVIEWED'"
        "Example: python train.py --asset-status-in TO_REVIEW,REVIEWED "
    ),
)
@click.option("--max-assets", default=None, type=int, help="Maximum number of assets to consider")
@click.option(
    "--model-name",
    default=None,
    help="Model name (one of efficientnet_b0, resnet50)",
)
@click.option("--project-id", default=None, required=True, help="Kili project ID")
@click.option("--verbose", default=0, type=int, help="Verbose level")
@click.option(
    "--disable-wandb",
    default=True,
    is_flag=True,
    help="Tells if wandb is disabled",
)
def main(
    api_endpoint: str,
    api_key: str,
    clear_dataset_cache: bool,
    model_framework: ModelFrameworkT,
    target_job: List[str],
    model_repository: ModelRepositoryT,
    dry_run: bool,
    epochs: int,
    asset_status_in: List[AssetStatusT],
    max_assets: int,
    model_name: ModelNameT,
    project_id: str,
    verbose: int,
    disable_wandb: bool,
    cv_folds: int,
):
    """
    Main method for detecting incorrect labeled assets in a Kili project.
    It downloads the assets, trains a classification neural network with CV and then by
    using the Cleanlab library we get the wrong labels. The concerned asset IDs are then
    stored in a file, but also a metadata (labeling_error: true) is uploaded to Kili to
    easily filter them later in the app.
    """

    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, _ = get_project(kili, project_id)

    for job_name, job in jobs.items():
        if target_job and job_name not in target_job:
            continue
        kili_print(f"Detecting errors for job: {job_name}")
        content_input = job.get("content", {}).get("input")
        ml_task: MLTaskT = job.get("mlTask")  # type: ignore
        if clear_dataset_cache:
            clear_automl_cache(
                command="label_errors",
                project_id=project_id,
                job_name=job_name,
                model_framework=model_framework,
                model_repository=model_repository,
            )

        if content_input == "radio" and input_type == "IMAGE" and ml_task == "CLASSIFICATION":

            assets = get_assets(
                kili,
                project_id,
                asset_status_in,
                max_assets=max_assets,
            )

            image_classification_model = PyTorchVisionImageClassificationModel(
                assets=assets,
                model_repository=model_repository,
                model_name=model_name,
                job_name=job_name,
                project_id=project_id,
                api_key=api_key,
            )

            found_errors = image_classification_model.find_errors(cv_folds, epochs, verbose)

            print()
            kili_print("Number of wrong labels found: ", len(found_errors))

            if found_errors:
                if not dry_run:
                    upload_errors_to_kili(found_errors, kili)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
