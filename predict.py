import os
from typing import Optional, List, Dict

import click
from kili.client import Kili

from utils.constants import (
    ContentInput,
    InputType,
    MLTask,
    ModelFramework,
    ModelRepository,
    Tool,
)
from utils.helpers import (
    JobPredictions,
    get_assets,
    get_project,
    kili_print,
    get_last_trained_model_path,
)


def predict_ner(
    api_key: str,
    assets: List[Dict],
    job_name: str,
    project_id: str,
    model_path: Optional[str],
    verbose: int,
) -> JobPredictions:
    model_path_res = get_last_trained_model_path(
        job_name=job_name,
        project_id=project_id,
        model_path=model_path,
        project_path_wildcard=["*", "model", "*", "*"],
        weights_filename="pytorch_model.bin",
    )
    split_path = os.path.normpath(model_path_res).split(os.path.sep)
    if split_path[-4] == ModelRepository.HuggingFace:
        model_repository = ModelRepository.HuggingFace
        kili_print(f"Model base repository: {model_repository}")
    else:
        raise ValueError("Unknown model base repository")
    if split_path[-2] in [ModelFramework.PyTorch, ModelFramework.Tensorflow]:
        model_framework = split_path[-2]
        kili_print(f"Model framework: {model_framework}")
    else:
        raise ValueError("Unknown model framework")
    if model_repository == ModelRepository.HuggingFace:
        from utils.huggingface.predict import huggingface_predict_ner

        return huggingface_predict_ner(
            api_key, assets, model_framework, model_path_res, job_name, verbose=verbose
        )
    else:
        raise NotImplementedError


def predict_object_detection(
    api_key: str,
    assets: List[Dict],
    job_name: str,
    project_id: str,
    model_path: Optional[str],
    verbose: int,
    prioritization: bool,
) -> JobPredictions:
    from utils.ultralytics.predict import ultralytics_predict_object_detection

    # /Users/raph/.cache/kili/automl/cl0wihlop3rwc0mtj9np28ti2/DETECTION/ultralytics/model/pytorch/2022-04-12_17_04_35/Severstal-steel-defect-detection
    model_path_res = get_last_trained_model_path(
        project_id=project_id,
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
    if model_repository not in [ModelRepository.Ultralytics]:
        raise ValueError(f"Unknown model base repository: {model_repository}")

    model_framework: ModelFramework = split_path[-5]  # type: ignore
    kili_print(f"Model framework: {model_framework}")
    if model_framework not in [ModelFramework.PyTorch, ModelFramework.Tensorflow]:
        raise ValueError(f"Unknown model framework: {model_framework}")

    if model_repository == ModelRepository.Ultralytics:
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
    project_id,
    from_model,
    verbose,
    input_type,
    assets,
    job_name,
    content_input,
    ml_task,
    tools,
    prioritization,
) -> JobPredictions:
    if (
        content_input == ContentInput.Radio
        and input_type == InputType.Text
        and ml_task == MLTask.NamedEntitiesRecognition
    ):
        job_predictions = predict_ner(
            api_key, assets, job_name, project_id, from_model, verbose=verbose
        )

    elif (
        content_input == ContentInput.Radio
        and input_type == InputType.Image
        and ml_task == MLTask.ObjectDetection
        and Tool.Rectangle in tools
    ):
        job_predictions = predict_object_detection(
            api_key, assets, job_name, project_id, from_model, verbose, prioritization
        )

    else:
        raise NotImplementedError
    return job_predictions


@click.command()
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
def main(
    api_key: str,
    project_id: str,
    label_types: str,
    dry_run: bool,
    from_model: Optional[ModelFramework],
    verbose: bool,
    max_assets: Optional[int],
):

    kili = Kili(api_key=api_key)
    input_type, jobs, _ = get_project(kili, project_id)
    assets = get_assets(kili, project_id, label_types.split(","), max_assets=max_assets)

    for job_name, job in jobs.items():
        content_input = job.get("content", {}).get("input")
        ml_task = job.get("mlTask")
        tools = job.get("tools")

        job_predictions = predict_one_job(
            api_key=api_key,
            project_id=project_id,
            from_model=from_model,
            verbose=verbose,
            input_type=input_type,
            assets=assets,
            job_name=job_name,
            content_input=content_input,
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
