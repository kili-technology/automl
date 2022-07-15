import os
from typing import List

import click
from kili.client import Kili

from commands.common_args import Options, TrainOptions
from kiliautoml.models import (
    Detectron2SemanticSegmentationModel,
    HuggingFaceNamedEntityRecognitionModel,
    HuggingFaceTextClassificationModel,
    PyTorchVisionImageClassificationModel,
    UltralyticsObjectDetectionModel,
)
from kiliautoml.models._base_model import BaseInitArgs, BaseTrainArgs
from kiliautoml.utils.helpers import (
    get_assets,
    get_content_input_from_job,
    get_project,
    is_contours_detection,
    kili_print,
    not_implemented_job,
    print_evaluation,
)
from kiliautoml.utils.memoization import clear_command_cache
from kiliautoml.utils.type import (
    AdditionalTrainingArgsT,
    AssetStatusT,
    JobNameT,
    LabelMergeStrategyT,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
    ToolT,
)


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
@Options.label_merge_strategy
@TrainOptions.asset_status_in
@TrainOptions.epochs
@TrainOptions.disable_wandb
@TrainOptions.additionalTrainArgsHuggingFace
@TrainOptions.additionalTrainArgsYolo
def main(
    api_endpoint: str,
    api_key: str,
    model_framework: ModelFrameworkT,
    model_name: ModelNameT,
    model_repository: ModelRepositoryT,
    project_id: ProjectIdT,
    epochs: int,
    asset_status_in: List[AssetStatusT],
    label_merge_strategy: LabelMergeStrategyT,
    target_job: List[JobNameT],
    max_assets: int,
    randomize_assets: bool,
    clear_dataset_cache: bool,
    disable_wandb: bool,
    verbose: int,
    batch_size: int,
    additional_train_args_hg: AdditionalTrainingArgsT,
    additional_train_args_yolo: AdditionalTrainingArgsT,
):
    """Train a model and then save the model in the cache.


    If there are multiple jobs in your projects, a model will be trained on each job.
    """
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, title = get_project(kili, project_id)

    model_evaluations = []

    for job_name, job in jobs.items():

        if target_job and job_name not in target_job:
            continue

        ml_task = job.get("mlTask")
        assets = get_assets(
            kili,
            project_id=project_id,
            status_in=asset_status_in,
            max_assets=max_assets,
            randomize=randomize_assets,
            strategy=label_merge_strategy,
            job_name=job_name,
        )

        kili_print(f"Training on job: {job_name}")
        os.environ["WANDB_PROJECT"] = title + "_" + job_name

        if clear_dataset_cache:
            clear_command_cache(
                command="train",
                project_id=project_id,
                job_name=job_name,
                model_framework=model_framework,
                model_repository=model_repository,
            )
        content_input = get_content_input_from_job(job)
        tools: List[ToolT] = job.get("tools")
        model_evaluation = {}

        base_init_args: BaseInitArgs = {
            "job": job,
            "job_name": job_name,
            "model_framework": model_framework,
            "model_name": model_name,
        }

        base_train_args = BaseTrainArgs(
            assets=assets,
            epochs=epochs,
            batch_size=batch_size,
            clear_dataset_cache=clear_dataset_cache,
            disable_wandb=disable_wandb,
            verbose=verbose,
        )
        if content_input == "radio" and input_type == "TEXT" and ml_task == "CLASSIFICATION":
            model = HuggingFaceTextClassificationModel(
                project_id=project_id,
                api_key=api_key,
                api_endpoint=api_endpoint,
                **base_init_args,
            )
            model_evaluation = model.train(
                **base_train_args, additional_train_args_hg=additional_train_args_hg
            )
        elif (
            content_input == "radio"
            and input_type == "TEXT"
            and ml_task == "NAMED_ENTITIES_RECOGNITION"
        ):
            model = HuggingFaceNamedEntityRecognitionModel(
                project_id=project_id,
                api_key=api_key,
                api_endpoint=api_endpoint,
                **base_init_args,
            )
            model_evaluation = model.train(
                **base_train_args, additional_train_args_hg=additional_train_args_hg
            )
        elif (
            content_input == "radio"
            and input_type == "IMAGE"
            and ml_task == "OBJECT_DETECTION"
            and "rectangle" in tools
        ):
            model = UltralyticsObjectDetectionModel(project_id=project_id, **base_init_args)
            model_evaluation = model.train(
                **base_train_args,
                title=title,  # TODO: delete
                api_key=api_key,  # TODO: Moove to initialisation
                additional_train_args_yolo=additional_train_args_yolo,
            )
        elif content_input == "radio" and input_type == "IMAGE" and ml_task == "CLASSIFICATION":
            image_classification_model = PyTorchVisionImageClassificationModel(
                model_repository=model_repository,  # TODO: delete
                project_id=project_id,
                **base_init_args,
            )
            model_evaluation = image_classification_model.train(**base_train_args, api_key=api_key)
        elif is_contours_detection(input_type, ml_task, content_input, tools):
            image_classification_model = Detectron2SemanticSegmentationModel(
                project_id=project_id, **base_init_args
            )
            model_evaluation = image_classification_model.train(
                **base_train_args,
                api_key=api_key,
                label_merge_strategy=label_merge_strategy,  # TODO: delete
                job=job,
            )

        else:
            not_implemented_job(job_name, ml_task, tools)

        model_evaluations.append((job_name, model_evaluation))

    kili_print()
    for job_name, evaluation in model_evaluations:
        print_evaluation(job_name, evaluation)
