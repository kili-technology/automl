import os
import json
import requests
import shutil
import time

import click
from kili.client import Kili
from tqdm.auto import tqdm

from utils.cleanlab.train_cleanlab import train_and_get_error_labels
from utils.constants import (
    HOME,
    ContentInput,
    InputType,
    MLTask,
    ModelName,
)
from utils.helpers import (
    build_model_repository_path,
    get_assets,
    get_project,
    kili_print,
    parse_label_types,
    set_default,
)


@click.command()
@click.option(
    "--api-endpoint",
    default="https://cloud.kili-technology.com/api/label/v2/graphql",
    help="Kili Endpoint",
)
@click.option("--api-key", default=os.environ.get("KILI_API_KEY"), help="Kili API Key")
@click.option("--cv-folds", default=4, type=int, help="Number of CV folds to use")
@click.option(
    "--clear-dataset-cache",
    default=False,
    is_flag=True,
    help="Tells if the dataset cache must be cleared",
)
@click.option(
    "--label-types",
    default="DEFAULT",
    help=(
        "Comma separated list Kili specific label types to select (among DEFAULT,"
        " REVIEW, PREDICTION)"
    ),
)
@click.option("--max-assets", default=None, type=int, help="Maximum number of assets to consider")
@click.option(
    "--model-name",
    default=None,
    help="Model name (one of efficientnet_b0, resnet50)",
)
@click.option("--project-id", default=None, help="Kili project ID")
@click.option(
    "--training-epochs",
    default=10,
    type=int,
    help="Number of epochs to train each CV fold",
)
def main(
    api_endpoint: str,
    api_key: str,
    cv_folds: int,
    clear_dataset_cache: bool,
    label_types: str,
    max_assets: int,
    model_name: str,
    project_id: str,
    training_epochs: int,
):
    """
    Main method for detecting incorrect labeled assets in a Kili project.
    It downloads the assets, trains a classification neural network with CV and then by
    using the Cleanlab library we get the wrong labels. The concerned asset IDs are then
    stored in a file, but also a metadata (cleanlab_error: true) is uploaded to Kili to
    easily filter them later in the app.
    """

    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, _ = get_project(kili, project_id)

    for job_name, job in jobs.items():
        content_input = job.get("content", {}).get("input")
        ml_task = job.get("mlTask")
        if (
            content_input == ContentInput.Radio
            and input_type == InputType.Image
            and ml_task == MLTask.Classification
        ):
            job_path = build_model_repository_path(HOME, project_id, job_name, "")
            data_path = os.path.join(job_path, "data")
            model_path = os.path.join(job_path, "model")

            if clear_dataset_cache and os.path.exists(job_path) and os.path.isdir(job_path):
                kili_print("Dataset cache for this project is being cleared.")
                shutil.rmtree(job_path)

            os.makedirs(job_path, exist_ok=True)
            os.makedirs(data_path, exist_ok=True)
            os.makedirs(model_path, exist_ok=True)

            kili_print("Downloading datasets from Kili")
            assets = get_assets(kili, project_id, parse_label_types(label_types), max_assets)
            if len(assets) == 0:
                raise Exception("No asset in dataset, exiting...")

            for asset in tqdm(assets):
                tic = time.time()
                n_try = 0
                while n_try < 20:
                    try:
                        img_data = requests.get(
                            asset["content"],
                            headers={
                                "Authorization": f"X-API-Key: {api_key}",
                            },
                        ).content
                        break
                    except Exception:
                        time.sleep(1)
                        n_try += 1
                img_path = os.path.join(
                    data_path, asset["labels"][0]["jsonResponse"][job_name]["categories"][0]["name"]
                )
                os.makedirs(img_path, exist_ok=True)
                with open(os.path.join(img_path, asset["id"] + ".jpg"), "wb") as handler:
                    handler.write(img_data)
                toc = time.time() - tic
                throttling_per_call = 60.0 / 250  # Kili API calls are limited to 250 per minute
                if toc < throttling_per_call:
                    time.sleep(throttling_per_call - toc)

            model_name = set_default(
                model_name,
                ModelName.EfficientNetB0,
                "model_name",
                [ModelName.EfficientNetB0, ModelName.Resnet50],
            )

            found_errors = train_and_get_error_labels(
                cv_n_folds=cv_folds,
                data_dir=data_path,
                model_dir=model_path,
                model_name=model_name,
                training_epochs=training_epochs,
            )

            print()
            kili_print("Number of wrong labels found: ", len(found_errors))

            if found_errors:
                found_errors_dict = {"assetIds": found_errors}
                found_errors_json = json.dumps(found_errors_dict, sort_keys=True, indent=4)
                if found_errors_json is not None:
                    json_path = os.path.join(job_path, "error_labels.json")
                    with open(json_path, "wb") as output_file:
                        output_file.write(found_errors_json.encode("utf-8"))
                        kili_print("Asset IDs of wrong labels written to: ", json_path)

                kili_print("Updating metadatas for the concerned assets")
                first = min(100, len(found_errors))
                for skip in tqdm(range(0, len(found_errors), first)):
                    error_assets = kili.assets(
                        asset_id_in=found_errors[skip : skip + first], fields=["id", "metadata"]
                    )
                    asset_ids = [asset["id"] for asset in error_assets]
                    new_metadatas = [asset["metadata"] for asset in error_assets]

                    for meta in new_metadatas:
                        meta["cleanlab_error"] = True

                    kili.update_properties_in_assets(
                        asset_ids=asset_ids, json_metadatas=new_metadatas
                    )
        else:
            kili_print("not implemented yet")


if __name__ == "__main__":
    main()
