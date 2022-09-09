from typing import List

import click
from kili.client import Kili

from commands.common_args import Options, EvaluateOptions
from kiliautoml.models._base_model import (
    BaseInitArgs,
    BaseEvaluateArgs,
    ModelEvaluateArgs,
    ModelConditionsRequested,
)
from kiliautoml.utils.helpers import (
    curated_job,
    get_assets,
    get_content_input_from_job,
    get_project,
    print_evaluation,
)
from kiliautoml.utils.logging import logger, set_kili_logging
from kiliautoml.utils.memoization import clear_command_cache
from kiliautoml.utils.type import (
    AdditionalTrainingArgsT,
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
@Options.clear_dataset_cache
@Options.batch_size
@Options.verbose
@Options.label_merge_strategy
@Options.parity_filter
@EvaluateOptions.asset_status_in
@EvaluateOptions.additionalTrainArgsHuggingFace
def main(
    api_endpoint: str,
    api_key: str,
    ml_backend: MLBackendT,
    model_name: ModelNameT,
    model_repository: ModelRepositoryT,
    project_id: ProjectIdT,
    asset_status_in: List[AssetStatusT],
    label_merge_strategy: LabelMergeStrategyT,
    target_job: List[JobNameT],
    ignore_job: List[JobNameT],
    max_assets: int,
    clear_dataset_cache: bool,
    verbose: VerboseLevelT,
    batch_size: int,
    parity_filter: ParityFilterT,
    additional_train_args_hg: AdditionalTrainingArgsT,
):
    """Evaluate trained model.

    If there are multiple jobs in your projects, it will evaluate each job independently.
    The model is then available for use by other commands such as `predict` and `label_errors`.
    """
    set_kili_logging(verbose)
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, title = get_project(kili, project_id)
    jobs = curated_job(jobs, target_job, ignore_job)

    model_evaluations = []

    for job_name, job in jobs.items():
        logger.info(f"Training on job: {job_name}")

        ml_task = job.get("mlTask")
        assets = get_assets(
            kili,
            project_id=project_id,
            status_in=asset_status_in,
            max_assets=max_assets,
            strategy=label_merge_strategy,
            job_name=job_name,
            parity_filter=parity_filter,
        )

        if clear_dataset_cache:
            clear_command_cache(
                command="evaluate",
                project_id=project_id,
                job_name=job_name,
                ml_backend=ml_backend,
                model_repository=model_repository,
            )
        content_input = get_content_input_from_job(job)
        tools: List[ToolT] = job.get("tools")
        model_evaluation = {}

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

        base_train_args = BaseEvaluateArgs(
            assets=assets,
            batch_size=batch_size,
            clear_dataset_cache=clear_dataset_cache,
        )

        model_train_args = ModelEvaluateArgs(
            additional_train_args_hg=additional_train_args_hg,
        )
        condition_requested = ModelConditionsRequested(
            input_type=input_type,
            ml_task=ml_task,
            content_input=content_input,
            ml_backend=ml_backend,
            model_name=model_name,
            model_repository=model_repository,
            tools=tools,
        )
        model = auto_get_instantiated_model(
            condition_requested=condition_requested,
            base_init_args=base_init_args,
        )
        model_evaluation = model.evaluate(
            base_train_args=base_train_args, model_train_args=model_train_args
        )

        model_evaluations.append((job_name, model_evaluation))

    logger.info("Summary of training:")
    for job_name, evaluation in model_evaluations:
        if evaluation is not None:
            print_evaluation(job_name, evaluation)

    logger.success("train command finished successfully!")
