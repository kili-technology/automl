import json
import os
from typing import List

import click
from kili.client import Kili
from tabulate import tabulate

from kiliautoml.models import (
    HuggingFaceNamedEntityRecognitionModel,
    HuggingFaceTextClassificationModel,
)
from kiliautoml.models._pytorchvision_image_classification import (
    PyTorchVisionImageClassificationModel,
)
from kiliautoml.utils.constants import (
    HOME,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
    ToolT,
)
from kiliautoml.utils.helpers import get_assets, get_project, kili_print, set_default
from kiliautoml.utils.memoization import clear_automl_cache
from kiliautoml.utils.path import Path
from kiliautoml.utils.type import AssetStatusT

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_image_bounding_box(
    *,
    api_key,
    job,
    job_name,
    max_assets,
    args_dict,
    epochs,
    model_framework,
    model_name,
    model_repository: ModelRepositoryT,
    project_id,
    asset_status_in,
    clear_dataset_cache,
    title,
    disable_wandb,
):
    from kiliautoml.utils.ultralytics.train import ultralytics_train_yolov5

    model_repository_initialized: ModelRepositoryT = set_default(
        model_repository,
        "ultralytics",
        "model_repository",
        ["ultralytics"],
    )
    path_repository = Path.model_repository_dir(
        HOME, project_id, job_name, model_repository_initialized
    )
    if model_repository_initialized == "ultralytics":
        model_framework = set_default(
            model_framework,
            "pytorch",
            "model_framework",
            ["pytorch"],
        )
        model_name = set_default(
            model_name, "ultralytics/yolov", "model_name", ["ultralytics/yolov"]
        )
        return ultralytics_train_yolov5(
            api_key=api_key,
            model_repository_dir=path_repository,
            job=job,
            max_assets=max_assets,
            json_args=args_dict,
            epochs=epochs,
            project_id=project_id,
            model_framework=model_framework,
            asset_status_in=asset_status_in,
            clear_dataset_cache=clear_dataset_cache,
            title=title,
            disable_wandb=disable_wandb,
        )
    else:
        raise NotImplementedError


@click.command()
@click.option(
    "--api-endpoint",
    default="https://cloud.kili-technology.com/api/label/v2/graphql",
    help="Kili Endpoint",
)
@click.option("--api-key", default=os.environ.get("KILI_API_KEY", ""), help="Kili API Key")
@click.option("--model-framework", default=None, help="Model framework (eg. pytorch, tensorflow)")
@click.option("--model-name", default=None, help="Model name (eg. bert-base-cased)")
@click.option("--model-repository", default=None, help="Model repository (eg. huggingface)")
@click.option("--project-id", required=True, help="Kili project ID")
@click.option(
    "--asset-status-in",
    default="LABELED,TO_REVIEW,REVIEWED",
    callback=lambda _, __, x: x.upper().split(",") if x else None,
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
            training_loss = HuggingFaceTextClassificationModel(
                project_id, api_key, api_endpoint
            ).train(
                assets=assets,
                job=job,
                job_name=job_name,
                model_framework=model_framework,
                clear_dataset_cache=clear_dataset_cache,
                epochs=epochs,
                disable_wandb=disable_wandb,
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

            training_loss = HuggingFaceNamedEntityRecognitionModel(
                project_id, api_key, api_endpoint
            ).train(
                assets=assets,
                job=job,
                job_name=job_name,
                model_framework=model_framework,
                clear_dataset_cache=clear_dataset_cache,
                epochs=epochs,
                disable_wandb=disable_wandb,
            )
        elif (
            content_input == "radio"
            and input_type == "IMAGE"
            and ml_task == "OBJECT_DETECTION"
            and "rectangle" in tools
        ):
            # no need to get_assets here because it's done in kili_template.yaml
            training_loss = train_image_bounding_box(
                api_key=api_key,
                job=job,
                job_name=job_name,
                max_assets=max_assets,
                args_dict=json.loads(json_args) if json_args is not None else {},
                model_framework=model_framework,
                model_name=model_name,
                model_repository=model_repository,
                project_id=project_id,
                epochs=epochs,
                asset_status_in=asset_status_in,
                clear_dataset_cache=clear_dataset_cache,
                title=title,
                disable_wandb=disable_wandb,
            )
        elif content_input == "radio" and input_type == "IMAGE" and ml_task == "CLASSIFICATION":

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

            training_loss = image_classification_model.train(epochs, verbose)

        else:
            kili_print("not implemented yet")
        training_losses.append([job_name, training_loss])
    kili_print()
    print(tabulate(training_losses, headers=["job_name", "training_loss"]))


if __name__ == "__main__":
    main()
