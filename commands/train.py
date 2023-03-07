import pathlib
from typing import List, Optional, cast

import click
from kili.client import Kili

from commands.common_args import EvaluateOptions, Options, TrainOptions
from kiliautoml.models._base_model import (
    BaseInitArgs,
    BaseTrainArgs,
    ModelConditionsRequested,
    ModelTrainArgs,
)
from kiliautoml.models.auto_get_model import auto_get_instantiated_model
from kiliautoml.utils.helpers import (
    curated_job,
    get_assets,
    get_content_input_from_job,
    get_project,
    print_and_save_evaluation,
)
from kiliautoml.utils.logging import logger, set_kili_logging
from kiliautoml.utils.memoization import clear_command_cache
from kiliautoml.utils.path import Path
from kiliautoml.utils.type import (
    AdditionalTrainingArgsT,
    AssetFilterArgsT,
    AssetStatusT,
    JobNameT,
    LabelMergeStrategyT,
    MLBackendT,
    ModelNameT,
    ModelRepositoryT,
    ParityFilterT,
    ProjectIdT,
    ToolT,
    VerboseLevelT,
)

import wandb  # isort:skip
from wandb.sdk.wandb_run import Run  # isort:skip


@click.command()
@Options.project_id
@Options.api_endpoint
@Options.api_key
@Options.ml_backend
@Options.model_name
@Options.model_repository
@Options.target_job
@Options.ignore_job
@Options.max_assets
@Options.local_dataset_dir
@Options.randomize_assets
@Options.clear_dataset_cache
@Options.batch_size
@Options.verbose
@Options.label_merge_strategy
@Options.parity_filter
@Options.asset_filter
@TrainOptions.asset_status_in
@TrainOptions.epochs
@TrainOptions.disable_wandb
@TrainOptions.additionalTrainArgsHuggingFace
@TrainOptions.additionalTrainArgsYolo
@EvaluateOptions.results_dir
def main(
    api_endpoint: str,
    api_key: str,
    ml_backend: MLBackendT,
    model_name: ModelNameT,
    model_repository: ModelRepositoryT,
    project_id: ProjectIdT,
    epochs: int,
    asset_status_in: List[AssetStatusT],
    label_merge_strategy: LabelMergeStrategyT,
    target_job: List[JobNameT],
    ignore_job: List[JobNameT],
    max_assets: int,
    randomize_assets: bool,
    clear_dataset_cache: bool,
    disable_wandb: bool,
    verbose: VerboseLevelT,
    batch_size: int,
    parity_filter: ParityFilterT,
    asset_filter: AssetFilterArgsT,
    additional_train_args_hg: AdditionalTrainingArgsT,
    additional_train_args_yolo: AdditionalTrainingArgsT,
    results_dir: Optional[str],
    local_dataset_dir: Optional[str],
):
    """Train a model and then save the model in the cache.


    If there are multiple jobs in your projects, a model will be trained on each job.
    KiliAutoML will automatically select the model types that match your Kili project's
    labeling tasks (image classification, text classification, image segmentation).
    Once the model is trained, the model is stored in the cache.
    The model is then available for use by other commands such as `predict` and `label_errors`.
    """
    set_kili_logging(verbose)
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, title = get_project(kili, project_id)
    jobs = curated_job(jobs, target_job, ignore_job)

    model_evaluations = []

    for job_name, job in jobs.items():
        ml_task = job["mlTask"]
        content_input = get_content_input_from_job(job)
        tools: List[ToolT] = job.get("tools", [])
        model_evaluation = {}

        condition_requested = ModelConditionsRequested(
            input_type=input_type,
            ml_task=ml_task,
            content_input=content_input,
            ml_backend=ml_backend,
            model_name=model_name,
            model_repository=model_repository,
            tools=tools,
        )

        logger.info(f"Training on job: {job_name}")
        assets = get_assets(
            kili,
            project_id=project_id,
            status_in=asset_status_in,
            max_assets=max_assets,
            randomize=randomize_assets,
            strategy=label_merge_strategy,
            job_name=job_name,
            parity_filter=parity_filter,
            asset_filter=asset_filter,
            query_content=local_dataset_dir is None,
        )

        if clear_dataset_cache:
            clear_command_cache(
                command="train",
                project_id=project_id,
                job_name=job_name,
                ml_backend=ml_backend,
                model_repository=model_repository,
            )

        wandb_run: Optional[Run] = None
        if not disable_wandb:
            wandb_run = cast(
                Run,
                wandb.init(  # type: ignore[attr-defined]
                    project=title + "_" + job_name,
                    reinit=True,
                    dir=Path.wandb_dir(project_id, job_name),
                ),
            )
        base_init_args = BaseInitArgs(
            job=job,
            job_name=job_name,
            model_name=model_name,
            project_id=project_id,
            ml_backend=ml_backend,
            api_key=api_key,
            api_endpoint=api_endpoint,
            title=title,
        )

        base_train_args = BaseTrainArgs(
            assets=assets,
            local_dataset_dir=pathlib.Path(local_dataset_dir) if local_dataset_dir else None,
            epochs=epochs,
            batch_size=batch_size,
            clear_dataset_cache=clear_dataset_cache,
            disable_wandb=disable_wandb,
        )

        model_train_args = ModelTrainArgs(
            additional_train_args_yolo=additional_train_args_yolo,
            additional_train_args_hg=additional_train_args_hg,
        )
        model = auto_get_instantiated_model(
            condition_requested=condition_requested,
            base_init_args=base_init_args,
        )
        model_evaluation = model.train(**base_train_args, model_train_args=model_train_args)

        if wandb_run is not None:
            wandb_run.finish()
        model_evaluations.append((job_name, model_evaluation))

    logger.info("Summary of training:")
    for job_name, evaluation in model_evaluations:
        print_and_save_evaluation(job_name, evaluation, results_dir)

    logger.success("train command finished successfully!")
