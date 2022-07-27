from typing import List, Optional

import click
from kili.client import Kili

from commands.common_args import Options, PredictOptions
from kiliautoml.models._base_model import (
    BaseInitArgs,
    BasePredictArgs,
    ModelConditionsRequested,
)
from kiliautoml.models.kili_auto_model import KiliAutoModel
from kiliautoml.utils.helpers import (
    curated_job,
    get_assets,
    get_content_input_from_job,
    get_project,
    kili_print,
)
from kiliautoml.utils.type import (
    AssetStatusT,
    JobNameT,
    MLBackendT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
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
@Options.randomize_assets
@Options.max_assets
@Options.clear_dataset_cache
@Options.batch_size
@Options.verbose
@PredictOptions.asset_status_in
@PredictOptions.from_model
@PredictOptions.from_project
@PredictOptions.dry_run
def main(
    project_id: ProjectIdT,
    api_endpoint: str,
    api_key: str,
    asset_status_in: List[AssetStatusT],
    target_job: List[JobNameT],
    ignore_job: List[JobNameT],
    dry_run: bool,
    from_model: Optional[str],
    verbose: bool,
    max_assets: Optional[int],
    randomize_assets: bool,
    from_project: Optional[ProjectIdT],
    model_name: Optional[ModelNameT],
    model_repository: Optional[ModelRepositoryT],
    ml_backend: MLBackendT,
    batch_size: int,
    clear_dataset_cache: bool,
):
    """Compute predictions and upload them to Kili."""
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, _ = get_project(kili, project_id)
    jobs = curated_job(jobs, target_job, ignore_job)

    assets = get_assets(
        kili, project_id, asset_status_in, max_assets=max_assets, randomize=randomize_assets
    )

    for job_name, job in jobs.items():
        kili_print(f"Predicting annotations for job: {job_name}")
        content_input = get_content_input_from_job(job)
        ml_task = job.get("mlTask")
        tools = job.get("tools")

        base_init_args = BaseInitArgs(
            job=job,
            job_name=job_name,
            model_name=model_name,
            project_id=project_id,
            ml_backend=ml_backend,
        )
        predict_args = BasePredictArgs(
            assets=assets,
            model_path=from_model,
            from_project=from_project,
            batch_size=batch_size,
            verbose=verbose,
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
        model = KiliAutoModel(
            condition_requested=condition_requested, base_init_args=base_init_args
        )
        job_predictions = model.predict(**predict_args)

        if not dry_run and job_predictions and job_predictions.external_id_array:
            kili.create_predictions(
                project_id,
                external_id_array=job_predictions.external_id_array,  # type:ignore
                json_response_array=job_predictions.json_response_array,
                model_name_array=job_predictions.model_name_array,
            )
            kili_print(
                "Predictions sent to kili, you can open the following url to check them out!"
            )
            status_filter = "%2C".join(asset_status_in)
            kili_print(
                f"{api_endpoint[:-21]}/label/projects/{project_id}/explore?statusIn={status_filter}"
            )
