from typing import List, Optional

import click
from kili.client import Kili

from commands.common_args import Options, PredictOptions
from kiliautoml.models import (
    HuggingFaceNamedEntityRecognitionModel,
    HuggingFaceTextClassificationModel,
    PyTorchVisionImageClassificationModel,
    UltralyticsObjectDetectionModel,
)
from kiliautoml.utils.constants import ModelFrameworkT
from kiliautoml.utils.helpers import JobPredictions, get_assets, get_project, kili_print
from kiliautoml.utils.type import AssetStatusT


def predict_one_job(
    *,
    api_key,
    api_endpoint,
    project_id,
    from_model,
    from_project: Optional[str],
    verbose,
    input_type,
    assets,
    job_name,
    batch_size,
    content_input,
    ml_task,
    model_repository,
    model_framework,
    model_name,
    tools,
    job,
    clear_dataset_cache,
) -> JobPredictions:
    if content_input == "radio" and input_type == "TEXT" and ml_task == "CLASSIFICATION":
        model = HuggingFaceTextClassificationModel(
            project_id,
            api_key,
            api_endpoint,
            model_name=model_name,
            job_name=job_name,
            job=job,
        )
        job_predictions = model.predict(
            assets=assets,
            model_path=from_model,
            batch_size=batch_size,
            verbose=verbose,
            from_project=from_project,
            clear_dataset_cache=clear_dataset_cache,
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
            model_name=model_name,
            job_name=job_name,
            job=job,
        )
        job_predictions = model.predict(
            assets=assets,
            model_path=from_model,
            batch_size=batch_size,
            verbose=verbose,
            from_project=from_project,
            clear_dataset_cache=clear_dataset_cache,
        )

    elif (
        content_input == "radio"
        and input_type == "IMAGE"
        and ml_task == "OBJECT_DETECTION"
        and "rectangle" in tools
    ):
        image_classification_model = UltralyticsObjectDetectionModel(
            job=job,
            model_framework=model_framework,
            model_name=model_name,
            job_name=job_name,
            project_id=project_id,
        )

        job_predictions = image_classification_model.predict(
            verbose=verbose,
            assets=assets,
            model_path=from_model,
            from_project=from_project,
            batch_size=batch_size,
            clear_dataset_cache=clear_dataset_cache,
            api_key=api_key,
        )
    elif content_input == "radio" and input_type == "IMAGE" and ml_task == "CLASSIFICATION":
        image_classification_model = PyTorchVisionImageClassificationModel(
            model_repository=model_repository,
            job=job,
            model_framework=model_framework,
            model_name=model_name,
            job_name=job_name,
            project_id=project_id,
        )

        job_predictions = image_classification_model.predict(
            verbose=verbose,
            assets=assets,
            model_path=from_model,
            from_project=from_project,
            batch_size=batch_size,
            clear_dataset_cache=clear_dataset_cache,
            api_key=api_key,
        )

    else:
        raise NotImplementedError
    return job_predictions


@click.command()
@Options.project_id
@Options.api_endpoint
@Options.api_key
@Options.model_framework
@Options.model_name
@Options.model_repository
@Options.target_job
@Options.max_assets
@Options.clear_dataset_cache
@Options.batch_size
@Options.verbose
@PredictOptions.asset_status_in
@PredictOptions.from_model
@PredictOptions.from_project
@PredictOptions.dry_run
def main(
    project_id: str,
    api_endpoint: str,
    api_key: str,
    asset_status_in: List[AssetStatusT],
    target_job: List[str],
    dry_run: bool,
    from_model: Optional[ModelFrameworkT],
    verbose: bool,
    max_assets: Optional[int],
    from_project: Optional[str],
    model_name: Optional[str],
    model_repository: Optional[str],
    model_framework: ModelFrameworkT,
    batch_size: int,
    clear_dataset_cache: bool,
):
    """After training, use this script to predict annotations suggestions
    on your remaining assets."""
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, _ = get_project(kili, project_id)
    assets = get_assets(kili, project_id, asset_status_in, max_assets=max_assets)

    for job_name, job in jobs.items():
        if target_job and job_name not in target_job:
            continue
        kili_print(f"Predicting annotations for job: {job_name}")
        content_input = job.get("content", {}).get("input")
        ml_task = job.get("mlTask")
        tools = job.get("tools")

        job_predictions = predict_one_job(
            api_key=api_key,
            api_endpoint=api_endpoint,
            project_id=project_id,
            from_model=from_model,
            verbose=verbose,
            job=job,
            input_type=input_type,
            assets=assets,
            batch_size=batch_size,
            job_name=job_name,
            content_input=content_input,
            model_repository=model_repository,
            model_name=model_name,
            model_framework=model_framework,
            from_project=from_project,
            ml_task=ml_task,
            tools=tools,
            clear_dataset_cache=clear_dataset_cache,
        )

        if not dry_run and job_predictions.external_id_array:
            kili.create_predictions(
                project_id,
                external_id_array=job_predictions.external_id_array,
                json_response_array=job_predictions.json_response_array,
                model_name_array=job_predictions.model_name_array,
            )
