from typing import List

import click
from kili.client import Kili

from commands.common_args import LabelErrorOptions, Options, TrainOptions
from kiliautoml.models._pytorchvision_image_classification import (
    PyTorchVisionImageClassificationModel,
)
from kiliautoml.utils.constants import (
    MLTaskT,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
)
from kiliautoml.utils.helpers import (
    get_labeled_assets,
    get_project,
    kili_print,
    upload_errors_to_kili,
)
from kiliautoml.utils.memoization import clear_automl_cache
from kiliautoml.utils.type import AssetStatusT, LabelMergeStrategyT


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
@Options.randomize_assets
@Options.batch_size
@Options.verbose
@TrainOptions.epochs
@LabelErrorOptions.asset_status_in
@Options.label_merge_strategy
@LabelErrorOptions.cv_folds
@LabelErrorOptions.dry_run
def main(
    project_id: str,
    api_endpoint: str,
    api_key: str,
    clear_dataset_cache: bool,
    model_framework: ModelFrameworkT,
    target_job: List[str],
    model_repository: ModelRepositoryT,
    dry_run: bool,
    epochs: int,
    asset_status_in: List[AssetStatusT],
    label_merge_strategy: LabelMergeStrategyT,
    max_assets: int,
    randomize_assets: bool,
    batch_size: int,
    model_name: ModelNameT,
    verbose: int,
    cv_folds: int,
):
    """
    Detect incorrectly labeled assets in a Kili project.

    It downloads the assets, trains a classification neural network with CV and then by
    using the Cleanlab library we get the wrong labels. The concerned asset IDs are then
    stored in a file, but also a metadata (labeling_error: true) is uploaded to Kili to
    easily filter them later in the app.
    """
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, _ = get_project(kili, project_id)

    for job_name, job in jobs.items():
        if target_job and job_name not in target_job:
            continue
        kili_print(f"Detecting errors for job: {job_name}")
        content_input = job.get("content", {}).get("input")
        ml_task: MLTaskT = job.get("mlTask")  # type: ignore
        if clear_dataset_cache:
            clear_automl_cache(
                command="label_errors",
                project_id=project_id,
                job_name=job_name,
                model_framework=model_framework,
                model_repository=model_repository,
            )

        assets = get_labeled_assets(
            kili,
            project_id=project_id,
            job_name=job_name,
            ml_task=ml_task,
            status_in=asset_status_in,
            max_assets=max_assets,
            randomize=randomize_assets,
            strategy=label_merge_strategy,
        )

        if content_input == "radio" and input_type == "IMAGE" and ml_task == "CLASSIFICATION":

            image_classification_model = PyTorchVisionImageClassificationModel(
                model_repository=model_repository,
                model_name=model_name,
                job_name=job_name,
                job=job,
                model_framework=model_framework,
                project_id=project_id,
            )
            found_errors = image_classification_model.find_errors(
                assets=assets,
                cv_n_folds=cv_folds,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                api_key=api_key,
            )

            print()
            kili_print("Number of wrong labels found: ", len(found_errors))

            if found_errors:
                if not dry_run:
                    upload_errors_to_kili(found_errors, kili)
        else:
            raise NotImplementedError
