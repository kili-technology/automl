from typing import List

import click
from kili.client import Kili

from commands.common_args import Options
from kiliautoml.models._base_model import ModelConditionsRequested
from kiliautoml.models.auto_get_model import auto_get_model_class
from kiliautoml.utils.helpers import (
    curated_job,
    get_content_input_from_job,
    get_project,
)
from kiliautoml.utils.logging import logger, set_kili_logging
from kiliautoml.utils.type import (
    JobNameT,
    MLBackendT,
    ModelNameT,
    ModelRepositoryT,
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
@Options.verbose
def main(
    api_endpoint: str,
    api_key: str,
    ml_backend: MLBackendT,
    model_name: ModelNameT,
    model_repository: ModelRepositoryT,
    project_id: ProjectIdT,
    target_job: List[JobNameT],
    ignore_job: List[JobNameT],
    verbose: VerboseLevelT,
):
    """Show models advised for each job of the project."""
    set_kili_logging(verbose)
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, _ = get_project(kili, project_id)
    jobs = curated_job(jobs, target_job, ignore_job)

    logger.info("Here are the advised models for your project:")
    for _, job in jobs.items():

        ml_task = job.get("mlTask")
        content_input = get_content_input_from_job(job)
        tools: List[ToolT] = job.get("tools")

        condition_requested = ModelConditionsRequested(
            input_type=input_type,
            ml_task=ml_task,
            content_input=content_input,
            ml_backend=ml_backend,
            model_name=model_name,
            model_repository=model_repository,
            tools=tools,
        )

        AppropriateModel = auto_get_model_class(condition_requested)

        advised_model_names = AppropriateModel.model_conditions.advised_model_names
        logger.info(f"For {ml_task} on {input_type}: {str(advised_model_names)}")

    logger.info(
        "For NLP tasks, you can also use any Fill-Mask model from Hugging Face. "
        "But be aware that some additional install might be necessary."
    )
    logger.info(
        "You can use the following command to train any of these compatible model:\n"
        "kiliautoml train --api-key $KILI_API_KEY --project-id $KILI_PROJECT_ID "
        "--model-name desired-model-name"
    )
