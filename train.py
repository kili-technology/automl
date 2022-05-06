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
from kiliautoml.utils.cleanlab.train_cleanlab import (
    train_and_get_error_image_classification,
)
from kiliautoml.utils.constants import (
    HOME,
    ContentInput,
    InputType,
    MLTask,
    ModelFramework,
    ModelFrameworkT,
    ModelName,
    ModelNameT,
    ModelRepository,
    ModelRepositoryT,
    Tool,
)
from kiliautoml.utils.helpers import (
    get_assets,
    get_project,
    kili_print,
    parse_label_types,
    set_default,
)
from kiliautoml.utils.memoization import clear_automl_cache
from kiliautoml.utils.path import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_image_bounding_box(
    *,
    api_key,
    job,
    job_name,
    max_assets,
    args_dict,
    model_framework,
    model_name,
    model_repository: ModelRepositoryT,
    project_id,
    label_types,
    clear_dataset_cache,
    title,
    disable_wandb,
):
    from kiliautoml.utils.ultralytics.train import ultralytics_train_yolov5

    model_repository_initialized: ModelRepositoryT = set_default(  # type: ignore
        model_repository,
        ModelRepository.Ultralytics,
        "model_repository",
        [ModelRepository.Ultralytics],
    )
    path_repository = Path.model_repository_dir(
        HOME, project_id, job_name, model_repository_initialized
    )
    if model_repository_initialized == ModelRepository.Ultralytics:
        model_framework = set_default(
            model_framework,
            ModelFramework.PyTorch,
            "model_framework",
            [ModelFramework.PyTorch],
        )
        model_name = set_default(model_name, ModelName.YoloV5, "model_name", [ModelName.YoloV5])
        return ultralytics_train_yolov5(
            api_key=api_key,
            model_repository_dir=path_repository,
            job=job,
            max_assets=max_assets,
            json_args=args_dict,
            project_id=project_id,
            model_framework=model_framework,
            label_types=label_types,
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
    "--label-types",
    default=None,
    help=(
        "Comma separated list Kili specific label types to select (among DEFAULT,"
        " REVIEW, PREDICTION)"
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
    label_types: str,
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
        tools = job.get("tools")
        training_loss = None
        if (
            content_input == ContentInput.Radio
            and input_type == InputType.Text
            and ml_task == MLTask.Classification
        ):

            assets = get_assets(
                kili,
                project_id,
                parse_label_types(label_types),
                labeling_statuses=["LABELED"],
            )
            assets = assets[:max_assets] if max_assets is not None else assets
            training_loss = HuggingFaceTextClassificationModel(
                project_id, api_key, api_endpoint
            ).train(
                assets=assets,
                job=job,
                job_name=job_name,
                model_framework=model_framework,
                model_name=model_name,
                clear_dataset_cache=clear_dataset_cache,
                disable_wandb=disable_wandb,
            )

        elif (
            content_input == ContentInput.Radio
            and input_type == InputType.Text
            and ml_task == MLTask.NamedEntityRecognition
        ):
            assets = get_assets(
                kili,
                project_id,
                parse_label_types(label_types),
                labeling_statuses=["LABELED"],
            )
            assets = assets[:max_assets] if max_assets is not None else assets

            training_loss = HuggingFaceNamedEntityRecognitionModel(
                project_id, api_key, api_endpoint
            ).train(
                assets=assets,
                job=job,
                job_name=job_name,
                model_framework=model_framework,
                model_name=model_name,
                clear_dataset_cache=clear_dataset_cache,
                disable_wandb=disable_wandb,
            )
        elif (
            content_input == ContentInput.Radio
            and input_type == InputType.Image
            and ml_task == MLTask.ObjectDetection
            and Tool.Rectangle in tools
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
                label_types=parse_label_types(label_types),
                clear_dataset_cache=clear_dataset_cache,
                title=title,
                disable_wandb=disable_wandb,
            )
        if (
            content_input == ContentInput.Radio
            and input_type == InputType.Image
            and ml_task == MLTask.Classification
        ):

            assets = get_assets(
                kili,
                project_id,
                parse_label_types(label_types),
                labeling_statuses=["LABELED"],
            )
            assets = assets[:max_assets] if max_assets is not None else assets

            training_loss = train_and_get_error_image_classification(
                cv_n_folds=None,
                epochs=epochs,
                job_name=job_name,
                model_repository=model_repository,
                project_id=project_id,
                assets=assets,
                model_name=model_name,
                api_key=api_key,
                verbose=verbose,
                only_train=True,
            )
        else:
            kili_print("not implemented yet")
        training_losses.append([job_name, training_loss])
    kili_print()
    print(tabulate(training_losses, headers=["job_name", "training_loss"]))


if __name__ == "__main__":
    main()
