import os
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
    save_errors,
    set_default,
    upload_errors_to_kili,
)


def download_assets(assets, api_key, data_path, job_name):
    """
    Download assets that are stored in Kili and save them to folders depending on their
    label category
    """
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


@click.command()
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
@click.option("--project-id", default=None, required=True, help="Kili project ID")
@click.option(
    "--training-epochs",
    default=10,
    type=int,
    help="Number of epochs to train each CV fold",
)
@click.option(
    "--upload-errors",
    default=True,
    type=bool,
    help="Upload 'labeling_error: True' metadata to Kili for concerned assets",
)
@click.option("--verbose", default=0, type=int, help="Verbose level")
def main(
    api_key: str,
    cv_folds: int,
    clear_dataset_cache: bool,
    label_types: str,
    max_assets: int,
    model_name: str,
    project_id: str,
    training_epochs: int,
    upload_errors: bool,
    verbose: int,
):
    """
    Main method for detecting incorrect labeled assets in a Kili project.
    It downloads the assets, trains a classification neural network with CV and then by
    using the Cleanlab library we get the wrong labels. The concerned asset IDs are then
    stored in a file, but also a metadata (labeling_error: true) is uploaded to Kili to
    easily filter them later in the app.
    """

    kili = Kili(api_key=api_key)
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

            download_assets(assets, api_key, data_path, job_name)

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
                verbose=verbose,
            )

            print()
            kili_print("Number of wrong labels found: ", len(found_errors))

            if found_errors:
                save_errors(found_errors, job_path)
                if upload_errors:
                    upload_errors_to_kili(found_errors, kili)
        else:
            kili_print("not implemented yet")


if __name__ == "__main__":
    main()
