from typing import List, Optional

import click
from kili.client import Kili

from commands.common_args import EvaluateOptions, Options
from kiliautoml.models._base_model import (
    BaseEvaluateArgs,
    BaseInitArgs,
    ModelConditionsRequested,
)
from kiliautoml.models.auto_get_model import auto_get_instantiated_model
from kiliautoml.utils.helpers import (
    curated_job,
    get_assets,
    get_content_input_from_job,
    get_project,
    print_evaluation,
)
from kiliautoml.utils.logging import logger, set_kili_logging
from kiliautoml.utils.type import (
    AssetStatusT,
    JobNameT,
    MLBackendT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
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
@EvaluateOptions.asset_status_in
@EvaluateOptions.model_path
def main(
    project_id: ProjectIdT,
    api_endpoint: str,
    api_key: str,
    ml_backend: MLBackendT,
    model_name: Optional[ModelNameT],
    model_repository: Optional[ModelRepositoryT],
    target_job: List[JobNameT],
    ignore_job: List[JobNameT],
    max_assets: Optional[int],
    clear_dataset_cache: bool,
    batch_size: int,
    verbose: VerboseLevelT,
    asset_status_in: List[AssetStatusT],
    model_path: Optional[str],
):
    """Compute predictions and upload them to Kili.

    In order to use this command, you must first use the `train` command.
    This command reuses the model that is stored at the end of the training.
    We advise you to use this command once in `--dry-run` mode to preview the predictions.
    Then, once you have validated the quality of the predictions, you can use this command without
    the `--dry-run` mode, which will directly upload the predictions to the online kili project.
    """
    set_kili_logging(verbose)
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, title = get_project(kili, project_id)
    jobs = curated_job(jobs, target_job, ignore_job)
    model_evaluations = []

    assets = get_assets(
        kili,
        project_id,
        asset_status_in,
        max_assets=max_assets,
    )

    for job_name, job in jobs.items():
        logger.info(f"Predicting annotations for job: {job_name}")
        content_input = get_content_input_from_job(job)
        ml_task = job.get("mlTask")
        tools = job.get("tools")

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
        evaluation_args = BaseEvaluateArgs(
            assets=assets,
            model_path=model_path,
            batch_size=batch_size,
            clear_dataset_cache=clear_dataset_cache,
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
            condition_requested=condition_requested, base_init_args=base_init_args
        )
        model_evaluation = model.eval(**evaluation_args)
        model_evaluations.append((job_name, model_evaluation))

    logger.info("Summary of evaluation:")
    for job_name, evaluation in model_evaluations:
        print_evaluation(job_name, evaluation)

    logger.success("Evaluate command finished successfully!")
