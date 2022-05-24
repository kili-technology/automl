import json
import os
from typing import List

import click
from kili.client import Kili
from tabulate import tabulate

from kiliautoml.models import (
    HuggingFaceNamedEntityRecognitionModel,
    HuggingFaceTextClassificationModel,
    PyTorchVisionImageClassificationModel,
    UltralyticsObjectDetectionModel,
)
from kiliautoml.utils.constants import (
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
    ToolT,
)
from kiliautoml.utils.helpers import get_assets, get_project, kili_print
from kiliautoml.utils.memoization import clear_automl_cache
from kiliautoml.utils.type import AssetStatusT


@click.command()
@click.option(
    "--api-endpoint",
    default="https://cloud.kili-technology.com/api/label/v2/graphql",
    help="Kili Endpoint",
)
@click.option("--api-key", default=os.environ.get("KILI_API_KEY", ""), help="Kili API Key")
@click.option(
    "--model-framework", default="pytorch", help="Model framework (eg. pytorch, tensorflow)"
)
@click.option("--model-name", default=None, help="Model name (eg. bert-base-cased)")
@click.option("--model-repository", default=None, help="Model repository (eg. huggingface)")
@click.option("--project-id", required=True, help="Kili project ID")
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
    "--epochs",
    default=10,
    type=int,
    show_default=True,
    help="Number of epochs to train for",
)
@click.option(
    "--max-assets",
    default=None,
    type=int,
    help="Maximum number of assets to consider",
)
@click.option(
    "--json-args",
    default=None,
    type=str,
    help=(
        "Specific parameters to pass to the trainer "
        "(for example Yolov5 train, Hugging Face transformers, ..."
    ),
)
@click.option(
    "--clear-dataset-cache",
    default=False,
    is_flag=True,
    help="Tells if the dataset cache must be cleared",
)
@click.option(
    "--disable-wandb",
    default=False,
    is_flag=True,
    help="Tells if wandb is disabled",
)
@click.option(
    "--batch-size",
    default=8,
    type=int,
    help="Maximum number of assets to consider",
)
@click.option("--verbose", default=0, type=int, help="Verbose level")
def main(
    api_endpoint: str,
    api_key: str,
    model_framework: ModelFrameworkT,
    model_name: ModelNameT,
    model_repository: ModelRepositoryT,
    project_id: str,
    epochs: int,
    asset_status_in: List[AssetStatusT],
    target_job: List[str],
    max_assets: int,
    json_args: str,
    clear_dataset_cache: bool,
    disable_wandb: bool,
    verbose: int,
    batch_size: int,
):
    """ """
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, title = get_project(kili, project_id)

    training_losses = []
    for job_name, job in jobs.items():
        if target_job and job_name not in target_job:
            continue
        kili_print(f"Training on job: {job_name}")
        os.environ["WANDB_PROJECT"] = title + "_" + job_name

        if clear_dataset_cache:
            clear_automl_cache(
                command="train",
                project_id=project_id,
                job_name=job_name,
                model_framework=model_framework,
                model_repository=model_repository,
            )
        content_input = job.get("content", {}).get("input")
        ml_task = job.get("mlTask")
        tools: List[ToolT] = job.get("tools")  # type: ignore
        training_loss = None
        if content_input == "radio" and input_type == "TEXT" and ml_task == "CLASSIFICATION":

            assets = get_assets(
                kili,
                project_id,
                asset_status_in,
                max_assets=max_assets,
            )

            model = HuggingFaceTextClassificationModel(
                project_id,
                api_key,
                api_endpoint,
                job=job,
                job_name=job_name,
                model_framework=model_framework,
            )

            training_loss = model.train(
                assets=assets,
                batch_size=batch_size,
                clear_dataset_cache=clear_dataset_cache,
                epochs=epochs,
                disable_wandb=disable_wandb,
                verbose=verbose,
            )

        elif (
            content_input == "radio"
            and input_type == "TEXT"
            and ml_task == "NAMED_ENTITIES_RECOGNITION"
        ):
            assets = get_assets(
                kili,
                project_id,
                asset_status_in,
                max_assets,
            )

            model = HuggingFaceNamedEntityRecognitionModel(
                project_id,
                api_key,
                api_endpoint,
                job=job,
                job_name=job_name,
                model_framework=model_framework,
            )

            training_loss = model.train(
                assets=assets,
                batch_size=batch_size,
                clear_dataset_cache=clear_dataset_cache,
                epochs=epochs,
                disable_wandb=disable_wandb,
                verbose=verbose,
            )
        elif (
            content_input == "radio"
            and input_type == "IMAGE"
            and ml_task == "OBJECT_DETECTION"
            and "rectangle" in tools
        ):
            assets = get_assets(
                kili,
                project_id,
                asset_status_in,
                max_assets,
            )

            model = UltralyticsObjectDetectionModel(
                project_id=project_id,
                job=job,
                job_name=job_name,
                model_framework=model_framework,
                model_name=model_name,
                model_repository=model_repository,
            )
            training_loss = model.train(
                assets=assets,
                epochs=epochs,
                batch_size=batch_size,
                clear_dataset_cache=clear_dataset_cache,
                disable_wandb=disable_wandb,
                title=title,
                args_dict=json.loads(json_args) if json_args is not None else {},
                api_key=api_key,
                verbose=verbose,
            )
        elif content_input == "radio" and input_type == "IMAGE" and ml_task == "CLASSIFICATION":

            assets = get_assets(
                kili,
                project_id,
                asset_status_in,
                max_assets=max_assets,
            )

            image_classification_model = PyTorchVisionImageClassificationModel(
                model_repository=model_repository,
                model_name=model_name,
                job=job,
                model_framework=model_framework,
                job_name=job_name,
                project_id=project_id,
            )

            training_loss = image_classification_model.train(
                assets=assets,
                batch_size=batch_size,
                epochs=epochs,
                clear_dataset_cache=clear_dataset_cache,
                disable_wandb=disable_wandb,
                api_key=api_key,
                verbose=verbose,
            )

        else:
            kili_print("not implemented yet")
        training_losses.append([job_name, training_loss])
    kili_print()
    print(tabulate(training_losses, headers=["job_name", "training_loss"]))


if __name__ == "__main__":
    main()
