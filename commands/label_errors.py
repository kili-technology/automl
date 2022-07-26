from typing import Any, Dict, List

import click
from kili.client import Kili
from tqdm.autonotebook import tqdm

from commands.common_args import LabelErrorOptions, Options, TrainOptions
from kiliautoml.models import (
    Detectron2SemanticSegmentationModel,
    PyTorchVisionImageClassificationModel,
    UltralyticsObjectDetectionModel,
)
from kiliautoml.models._base_model import BaseInitArgs
from kiliautoml.utils.helper_label_error import ErrorRecap, LabelingError
from kiliautoml.utils.helpers import (
    get_assets,
    get_content_input_from_job,
    get_project,
    is_contours_detection,
    kili_print,
    not_implemented_job,
)
from kiliautoml.utils.memoization import clear_command_cache
from kiliautoml.utils.type import (
    AssetStatusT,
    JobNameT,
    LabelMergeStrategyT,
    MLBackendT,
    MLTaskT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
    ToolT,
)


def upload_errors_to_kili(error_recap: ErrorRecap, kili: Kili, project_id: ProjectIdT):
    kili_print("Updating metadatas for the concerned assets")

    print(error_recap)
    found_errors = [asset_error for asset_error in error_recap if asset_error]
    kili_print("Number of wrong labels found: ", len(found_errors))

    asset_bundle = list(zip(error_recap.id_array, error_recap.errors_by_asset))
    first = min(100, len(asset_bundle))
    for skip in tqdm(
        range(0, len(asset_bundle), first),
        desc="Updating asset metadata with labeling error flag",
    ):
        assets = asset_bundle[skip : skip + first]
        errors_by_asset = [a[1] for a in assets]
        asset_ids = [a[0] for a in assets]

        asset_id_chunk = asset_ids[skip : skip + first]
        error_assets: List[Dict[str, str]] = kili.assets(
            asset_id_in=asset_id_chunk,  # type:ignore
            fields=["id", "metadata"],
            project_id=project_id,
        )
        asset_ids = [asset["id"] for asset in error_assets]
        metadatas = [asset["metadata"] for asset in error_assets]

        for i, (meta, errors) in enumerate(zip(metadatas, errors_by_asset)):
            update_asset_metadata(
                meta,  # type: ignore
                errors,
            )
            metadatas[i] = meta

        kili.update_properties_in_assets(
            asset_ids=asset_ids,
            json_metadatas=metadatas,  # type:ignore
        )


def update_asset_metadata(meta: Dict[str, Any], errors: List[LabelingError]):
    if len(errors):
        meta["labeling_error"] = "True"
        # Here we should return the main error
        meta["error_type"] = str(errors[0].error_type)
        meta["error_probability"] = str(errors[0].error_probability)
    else:
        meta.pop("labeling_error", None)
        meta.pop("error_type", None)
        meta.pop("error_probability", None)


@click.command()
@Options.project_id
@Options.api_endpoint
@Options.api_key
@Options.ml_backend
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
    project_id: ProjectIdT,
    api_endpoint: str,
    api_key: str,
    clear_dataset_cache: bool,
    ml_backend: MLBackendT,
    target_job: List[JobNameT],
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
        content_input = get_content_input_from_job(job)
        ml_task: MLTaskT = job.get("mlTask")  # type: ignore
        tools: List[ToolT] = job.get("tools")

        # We should delete ml_backend
        if clear_dataset_cache:
            clear_command_cache(
                command="label_errors",
                project_id=project_id,
                job_name=job_name,
                ml_backend=ml_backend,
                model_repository=model_repository,
            )

        assets = get_assets(
            kili,
            project_id=project_id,
            status_in=asset_status_in,
            max_assets=max_assets,
            randomize=randomize_assets,
            strategy=label_merge_strategy,
            job_name=job_name,
        )

        base_init_args: BaseInitArgs = {
            "job": job,
            "job_name": job_name,
            "model_name": model_name,
            "project_id": project_id,
        }

        if content_input == "radio" and input_type == "IMAGE" and ml_task == "CLASSIFICATION":

            image_classification_model = PyTorchVisionImageClassificationModel(**base_init_args)
            found_errors = image_classification_model.find_errors(
                assets=assets,
                cv_n_folds=cv_folds,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                api_key=api_key,
            )

        elif (
            content_input == "radio"
            and input_type == "IMAGE"
            and ml_task == "OBJECT_DETECTION"
            and "rectangle" in tools
        ):

            model = UltralyticsObjectDetectionModel(**base_init_args)
            found_errors = model.find_errors(
                cv_n_folds=cv_folds,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                api_key=api_key,
                assets=assets,
                clear_dataset_cache=clear_dataset_cache,
            )
        elif is_contours_detection(input_type, ml_task, content_input, tools):
            model = Detectron2SemanticSegmentationModel(**base_init_args)
            found_errors = model.find_errors(
                cv_n_folds=cv_folds,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                api_key=api_key,
                assets=assets,
                clear_dataset_cache=clear_dataset_cache,
            )
        else:
            not_implemented_job(job_name, ml_task, tools)
            raise Exception("Not implemented label_error MLtask.")

        if found_errors:
            if not dry_run:
                upload_errors_to_kili(found_errors, kili, project_id)
