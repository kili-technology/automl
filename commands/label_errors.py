from typing import Any, Dict, List

import click
from kili.client import Kili
from tqdm.autonotebook import tqdm
from typing_extensions import get_args

from commands.common_args import LabelErrorOptions, Options, TrainOptions
from kiliautoml.models import (
    Detectron2SemanticSegmentationModel,
    PyTorchVisionImageClassificationModel,
    UltralyticsObjectDetectionModel,
)
from kiliautoml.models._base_model import (
    BaseInitArgs,
    BaseLabelErrorsArgs,
    ModelConditionsRequested,
)
from kiliautoml.models.kili_auto_model import KiliAutoModel
from kiliautoml.utils.helper_label_error import (
    ErrorRecap,
    LabelingError,
    LabelingErrorTypeT,
)
from kiliautoml.utils.helpers import (
    curated_job,
    dry_run_security,
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
    ModelNameT,
    ModelRepositoryT,
    ParityFilterT,
    ProjectIdT,
)


def upload_errors_to_kili(error_recap: ErrorRecap, kili: Kili, project_id: ProjectIdT):
    kili_print("\nUpdating metadata for the concerned assets")

    found_errors = [len(asset_error) for asset_error in error_recap.errors_by_asset]
    kili_print("Number of wrong labels found: ", sum(found_errors))

    id_errors_tuples = list(zip(error_recap.id_array, error_recap.errors_by_asset))
    first = min(100, len(id_errors_tuples))
    for skip in tqdm(
        range(0, len(id_errors_tuples), first),
        desc="Updating asset metadata with labeling error flag",
    ):
        ids_errors = id_errors_tuples[skip : skip + first]
        errors_by_asset = [a[1] for a in ids_errors]
        asset_ids = [a[0] for a in ids_errors]

        # *_set means shuffled
        error_assets_set: List[Dict[str, str]] = kili.assets(
            asset_id_in=asset_ids,  # type:ignore
            fields=["id", "metadata"],
            project_id=project_id,
        )
        asset_ids_set = [asset["id"] for asset in error_assets_set]

        # unshuffle
        error_assets = [error_assets_set[asset_ids_set.index(id)] for id in asset_ids]
        metadata = [asset["metadata"] for asset in error_assets]

        for i, (meta, errors) in enumerate(zip(metadata, errors_by_asset)):
            update_asset_metadata(
                meta,  # type: ignore
                errors,
            )
            metadata[i] = meta

        kili.update_properties_in_assets(
            asset_ids=asset_ids,  # type:ignore
            json_metadata=metadata,  # type:ignore
        )


def update_asset_metadata(meta: Dict[str, Any], errors: List[LabelingError]):
    for error_type in get_args(LabelingErrorTypeT):
        meta.pop(f"has_{error_type}", None)
    meta.pop("error_labeling", None)
    meta.pop("error_type", None)
    meta.pop("error_probability", None)
    meta.pop("error_asset_detail", None)

    if len(errors) == 0:
        return

    meta["error_labeling"] = "True"

    # We can only have one error type by asset
    main_error = max(errors)
    meta["error_type"] = str(main_error.error_type)
    meta["error_probability"] = str(main_error.error_probability)

    # Getting the details
    mapping_error_cat_to_nb = {error.error_type: 0 for error in errors}
    for error in errors:
        mapping_error_cat_to_nb[error.error_type] += 1
        meta[f"has_{error.error_type}"] = str(True)
    meta["error_asset_detail"] = str(mapping_error_cat_to_nb)


def label_error(
    *,
    api_key,
    clear_dataset_cache,
    epochs,
    batch_size,
    verbose,
    cv_folds,
    input_type,
    job_name,
    content_input,
    ml_task,
    tools,
    assets,
    base_init_args,
):
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
            assets=assets,
            clear_dataset_cache=clear_dataset_cache,
        )
    else:
        not_implemented_job(job_name, ml_task, tools)
        raise Exception(
            f"Not implemented label_error MLtask : {ml_task} for job {job_name}. Please use"
            " --target-job XXX"
        )

    return found_errors


@click.command()
@Options.project_id
@Options.api_endpoint
@Options.api_key
@Options.ml_backend
@Options.model_name
@Options.model_repository
@Options.target_job
@Options.ignore_job
@Options.max_assets
@Options.clear_dataset_cache
@Options.randomize_assets
@Options.batch_size
@Options.verbose
@Options.parity_filter
@Options.label_merge_strategy
@TrainOptions.epochs
@LabelErrorOptions.asset_status_in
@LabelErrorOptions.cv_folds
@LabelErrorOptions.dry_run
@LabelErrorOptions.erase_error_metadata
def main(
    project_id: ProjectIdT,
    api_endpoint: str,
    api_key: str,
    clear_dataset_cache: bool,
    ml_backend: MLBackendT,
    target_job: List[JobNameT],
    ignore_job: List[JobNameT],
    model_repository: ModelRepositoryT,
    dry_run: bool,
    epochs: int,
    asset_status_in: List[AssetStatusT],
    label_merge_strategy: LabelMergeStrategyT,
    max_assets: int,
    randomize_assets: bool,
    batch_size: int,
    model_name: ModelNameT,
    parity_filter: ParityFilterT,
    verbose: int,
    cv_folds: int,
    erase_error_metadata: bool,
):
    """
    Detect incorrectly labeled assets in a Kili project.

    It downloads the assets, trains a classification neural network with CV and then by
    using the Cleanlab library we get the wrong labels. The concerned asset IDs are then
    stored in a file, but also a metadata (labeling_error: true) is uploaded to Kili to
    easily filter them later in the app.
    """
    dry_run = dry_run_security(dry_run)
    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, title = get_project(kili, project_id)
    jobs = curated_job(jobs, target_job, ignore_job)

    for job_name, job in jobs.items():
        kili_print(f"Detecting errors for job: {job_name}")
        content_input = get_content_input_from_job(job)
        ml_task = job.get("mlTask")
        tools = job.get("tools")

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
            parity_filter=parity_filter,
        )

        base_init_args = BaseInitArgs(
            job=job,
            job_name=job_name,
            model_name=model_name,
            project_id=project_id,
            ml_backend=ml_backend,
            api_key=api_key,
            api_endpoint=api_endpoint,
            title=title,
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

        base_label_errors_args = BaseLabelErrorsArgs(
            cv_n_folds=cv_folds,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            assets=assets,
            clear_dataset_cache=clear_dataset_cache,
        )

        empty_errors_recap = ErrorRecap(
            external_id_array=[a.externalId for a in assets],
            id_array=[a.id for a in assets],
            errors_by_asset=[[] for _ in assets],
        )
        model = KiliAutoModel(
            base_init_args=base_init_args, condition_requested=condition_requested
        )
        found_errors = (
            model.find_errors(base_label_errors_args=base_label_errors_args)
            if not erase_error_metadata
            else empty_errors_recap
        )

        if found_errors:
            if not dry_run:
                upload_errors_to_kili(found_errors, kili, project_id)
