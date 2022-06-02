import json
import os
from typing import List

import click
from kili.client import Kili
from tabulate import tabulate

from commands.common_args import Options, TrainOptions
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
from kiliautoml.utils.type import AssetStatusT, LabelMergeStrategyT


@click.command()
@Options.project_id
@Options.api_endpoint
@Options.api_key
@Options.model_framework
@Options.model_name
@Options.model_repository
@Options.target_job
@Options.max_assets
@Options.randomize_assets
@Options.clear_dataset_cache
@Options.batch_size
@Options.verbose
@TrainOptions.asset_status_in
@TrainOptions.label_merge_strategy
@TrainOptions.epochs
@TrainOptions.json_args
@TrainOptions.disable_wandb
def main(
    api_endpoint: str,
    api_key: str,
    model_framework: ModelFrameworkT,
    model_name: ModelNameT,
    model_repository: ModelRepositoryT,
    project_id: str,
    epochs: int,
    asset_status_in: List[AssetStatusT],
    label_merge_strategy: LabelMergeStrategyT,
    target_job: List[str],
    max_assets: int,
    randomize_assets: bool,
    json_args: str,
    clear_dataset_cache: bool,
    disable_wandb: bool,
    verbose: int,
    batch_size: int,
):
    """Train a model and then save the model in the cache.


    If there are multiple jobs in your projects, a model will be trained on each job.
    """
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, title = get_project(kili, project_id)

    training_losses = []

    assets = get_assets(
        kili, project_id, asset_status_in, max_assets=max_assets, randomize=randomize_assets
    )
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
                label_merge_strategy=label_merge_strategy,
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
                label_merge_strategy=label_merge_strategy,
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

            model = UltralyticsObjectDetectionModel(
                project_id=project_id,
                job=job,
                job_name=job_name,
                model_framework=model_framework,
                model_name=model_name,
            )
            training_loss = model.train(
                assets=assets,
                label_merge_strategy=label_merge_strategy,
                epochs=epochs,
                batch_size=batch_size,
                clear_dataset_cache=clear_dataset_cache,
                disable_wandb=disable_wandb,
                title=title,
                json_args=json.loads(json_args) if json_args is not None else {},
                api_key=api_key,
                verbose=verbose,
            )
        elif content_input == "radio" and input_type == "IMAGE" and ml_task == "CLASSIFICATION":

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
