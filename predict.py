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
):
    model_path_res = get_last_trained_model_path(
        job_name, project_id, model_path, ["*", "model", "*", "*"], "pytorch_model.bin"
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


def predict_object_detection(
    api_key: str,
    assets: List[Dict],
    job_name: str,
    project_id: str,
    model_path: Optional[str],
    verbose: int,
):
    from utils.ultralytics.predict import ultralytics_predict_object_detection

    model_path_res = get_last_trained_model_path(
        job_name,
        project_id,
        model_path,
        ["*", "model", "*", "*", "exp", "weights"],
        "best.pt",
    )
    split_path = os.path.normpath(model_path_res).split(os.path.sep)
    if split_path[-6] == ModelRepository.Ultralytics:
        model_repository = ModelRepository.Ultralytics
        kili_print(f"Model base repository: {model_repository}")
    else:
        raise ValueError(f"Unknown model base repository: {model_repository}")
    if split_path[-4] in [ModelFramework.PyTorch, ModelFramework.Tensorflow]:
        model_framework = split_path[-4]
        kili_print(f"Model framework: {model_framework}")
    else:
        raise ValueError(f"Unknown model framework: {model_framework}")
    if model_repository == ModelRepository.Ultralytics:
        return ultralytics_predict_object_detection(
            api_key,
            assets,
            project_id,
            model_framework,
            model_path_res,
            job_name,
            verbose,
        )


@click.command()
@click.option("--api-key", default=os.environ["KILI_API_KEY"], help="Kili API Key")
@click.option("--project-id", required=True, help="Kili project ID")
@click.option(
    "--label-types",
    default="DEFAULT,REVIEW",
    help="Comma separated list Kili specific label types to select (among DEFAULT, REVIEW, PREDICTION), defaults to 'DEFAULT,REVIEW'",
)
@click.option(
    "--dry-run",
    default=None,
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
    from_model: Optional[str],
    verbose: bool,
    max_assets: Optional[int],
):

    kili = Kili(api_key=api_key)
    input_type, jobs = get_project(kili, project_id)
    assets = get_assets(
        kili,
        project_id,
        label_types.split(","),
        max_assets=max_assets,
        only_labeled=True,
    )

    for job_name, job in jobs.items():
        content_input = job.get("content", {}).get("input")
        ml_task = job.get("mlTask")
        tools = job.get("tools")

        if (
            content_input == ContentInput.Radio
            and input_type == InputType.Text
            and ml_task == MLTask.NamedEntitiesRecognition
        ):
            json_responses = predict_ner(
                api_key, assets, job_name, project_id, from_model, verbose=verbose
            )

            if not dry_run:
                kili.create_predictions(
                    project_id,
                    external_id_array=[a["externalId"] for a in assets],
                    json_response_array=json_responses,
                    model_name_array=["Kili AutoML"] * len(assets),
                )

        elif (
            content_input == ContentInput.Radio
            and input_type == InputType.Image
            and ml_task == MLTask.ObjectDetection
            and Tool.Rectangle in tools
        ):
            id_json_list = predict_object_detection(
                api_key, assets, job_name, project_id, from_model, verbose
            )

            if not dry_run:
                kili.create_predictions(
                    project_id,
                    external_id_array=[a[0] for a in id_json_list],
                    json_response_array=[a[1] for a in id_json_list],
                    model_name_array=["Kili AutoML"] * len(id_json_list),
                )

        else:
            kili_print("not implemented yet")


if __name__ == "__main__":
    main()
