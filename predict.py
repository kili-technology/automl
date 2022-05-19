import os
from typing import List, Optional

import click
from kili.client import Kili

from kiliautoml.models import (
    HuggingFaceNamedEntityRecognitionModel,
    HuggingFaceTextClassificationModel,
)
from kiliautoml.models._pytorchvision_image_classification import (
    PyTorchVisionImageClassificationModel,
)
from kiliautoml.utils.constants import ModelFrameworkT
from kiliautoml.utils.helpers import (
    JobPredictions,
    get_assets,
    get_last_trained_model_path,
    get_project,
    kili_print,
)
from kiliautoml.utils.type import AssetT, LabelTypeT


def predict_object_detection(
    *,
    api_key: str,
    assets: List[AssetT],
    job_name: str,
    project_id: str,
    model_path: Optional[str],
    verbose: int,
    prioritization: bool,
    from_project: Optional[str],
) -> JobPredictions:
    from kiliautoml.utils.ultralytics.predict_ultralytics import (
        ultralytics_predict_object_detection,
    )

    model_path_res = get_last_trained_model_path(
        project_id=project_id if from_project is None else from_project,
        job_name=job_name,
        project_path_wildcard=[
            "*",  # ultralytics or huggingface
            "model",
            "*",  # pytorch or tensorflow
            "*",  # date and time
            "*",  # title of the project, but already specified by project_id
            "exp",
            "weights",
        ],
        weights_filename="best.pt",
        model_path=model_path,
    )
    split_path = os.path.normpath(model_path_res).split(os.path.sep)
    model_repository = split_path[-7]
    kili_print(f"Model base repository: {model_repository}")
    if model_repository not in ["ultralytics"]:
        raise ValueError(f"Unknown model base repository: {model_repository}")

    model_framework: ModelFrameworkT = split_path[-5]  # type: ignore
    kili_print(f"Model framework: {model_framework}")
    if model_framework not in ["pytorch", "tensorflow"]:
        raise ValueError(f"Unknown model framework: {model_framework}")

    if model_repository == "ultralytics":
        job_predictions = ultralytics_predict_object_detection(
            api_key,
            assets,
            project_id,
            model_framework,
            model_path_res,
            job_name,
            verbose,
            prioritization,
        )
    else:
        raise NotImplementedError

    return job_predictions


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
    content_input,
    ml_task,
    model_repository,
    model_name,
    tools,
    prioritization,
) -> JobPredictions:
    if content_input == "radio" and input_type == "TEXT" and ml_task == "CLASSIFICATION":
        job_predictions = HuggingFaceTextClassificationModel(
            project_id,
            api_key,
            api_endpoint,
            model_name=model_name,
        ).predict(
            assets=assets,
            job_name=job_name,
            model_path=from_model,  # TODO: rename from_model to model_path
            from_project=from_project,
            verbose=verbose,
        )

    elif (
        content_input == "radio"
        and input_type == "TEXT"
        and ml_task == "NAMED_ENTITIES_RECOGNITION"
    ):
        job_predictions = HuggingFaceNamedEntityRecognitionModel(
            project_id,
            api_key,
            api_endpoint,
            model_name=model_name,
        ).predict(
            assets=assets,
            job_name=job_name,
            model_path=from_model,
            verbose=verbose,
            from_project=from_project,
        )

    elif (
        content_input == "radio"
        and input_type == "IMAGE"
        and ml_task == "OBJECT_DETECTION"
        and "rectangle" in tools
    ):
        job_predictions = predict_object_detection(
            api_key=api_key,
            assets=assets,
            job_name=job_name,
            project_id=project_id,
            model_path=from_model,
            verbose=verbose,
            prioritization=prioritization,
            from_project=from_project,
        )
    elif content_input == "radio" and input_type == "IMAGE" and ml_task == "CLASSIFICATION":
        image_classification_model = PyTorchVisionImageClassificationModel(
            assets=assets,
            model_repository=model_repository,
            model_name=model_name,
            job_name=job_name,
            project_id=project_id,
            api_key=api_key,
        )

        job_predictions = image_classification_model.predict(verbose, assets, job_name)

    else:
        raise NotImplementedError
    return job_predictions


@click.command()
@click.option(
    "--api-endpoint",
    default="https://cloud.kili-technology.com/api/label/v2/graphql",
    help="Kili Endpoint",
)
@click.option("--api-key", default=os.environ.get("KILI_API_KEY"), help="Kili API Key")
@click.option("--project-id", required=True, help="Kili project ID")
@click.option(
    "--label-types",
    default="DEFAULT,REVIEW",
    help=(
        "Comma separated list Kili specific label types to select (among DEFAULT,"
        " REVIEW, PREDICTION), defaults to 'DEFAULT,REVIEW'"
    ),
)
@click.option("--model-name", default=None, help="Model name (eg. bert-base-cased)")
@click.option("--model-repository", default=None, help="Model repository (eg. huggingface)")
@click.option(
    "--target-job",
    default=None,
    multiple=True,
    help=(
        "Add a specific target job for which to output the predictions "
        "(multiple can be passed if --target-job <job_name> is repeated) "
        "Example: python predict.py --target-job BBOX --target-job CLASSIFICATION"
    ),
)
@click.option(
    "--dry-run",
    default=False,
    is_flag=True,
    help="Runs the predictions but do not save them into the Kili project",
)
@click.option(
    "--from-model",
    default=None,
    help="Runs the predictions using a specified model path",
)
@click.option(
    "--verbose",
    default=0,
    help="Verbose level",
)
@click.option(
    "--max-assets",
    default=None,
    type=int,
    help="Maximum number of assets to consider",
)
@click.option(
    "--from-project",
    default=None,
    type=str,
    help=(
        "Use a model trained of a different project to predict on project_id."
        "This is usefull if you do not want to pollute the original project with "
        "experimental predictions."
        "This argument is ignored if --from-model is used."
    ),
)
def main(
    api_endpoint: str,
    api_key: str,
    project_id: str,
    label_types: str,
    target_job: List[str],
    dry_run: bool,
    from_model: Optional[ModelFrameworkT],
    verbose: bool,
    max_assets: Optional[int],
    from_project: Optional[str],
    model_name: Optional[str],
    model_repository: Optional[str],
):
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, _ = get_project(kili, project_id)
    label_type_in: List[LabelTypeT] = label_types.split(",")  # type: ignore
    assets = get_assets(kili, project_id, label_type_in, max_assets=max_assets)
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
            input_type=input_type,
            assets=assets,
            job_name=job_name,
            content_input=content_input,
            model_repository=model_repository,
            model_name=model_name,
            from_project=from_project,
            ml_task=ml_task,
            tools=tools,
            prioritization=False,
        )

        if not dry_run and job_predictions.external_id_array:
            kili.create_predictions(
                project_id,
                external_id_array=job_predictions.external_id_array,
                json_response_array=job_predictions.json_response_array,
                model_name_array=job_predictions.model_name_array,
            )


if __name__ == "__main__":
    main()
