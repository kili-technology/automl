import warnings
from typing import List, Optional

import click
import pandas as pd
from kili.client import Kili
from tabulate import tabulate

from commands.common_args import (
    LabelErrorOptions,
    Options,
    PredictOptions,
    TrainOptions,
)
from commands.predict import predict_one_job
from kiliautoml.models import PyTorchVisionImageClassificationModel
from kiliautoml.utils.constants import ModelFrameworkT, ModelNameT, ModelRepositoryT
from kiliautoml.utils.helpers import get_assets, get_label, get_project, kili_print
from kiliautoml.utils.mapper.create import MapperClassification
from kiliautoml.utils.type import AssetStatusT, LabelMergeStrategyT


@click.command()
@Options.api_endpoint
@Options.api_key
@Options.project_id
@Options.clear_dataset_cache
@Options.batch_size
@TrainOptions.epochs
@Options.target_job
@Options.model_framework
@Options.model_name
@Options.model_repository
@TrainOptions.asset_status_in
@Options.label_merge_strategy
@Options.max_assets
@click.option(
    "--assets-repository",
    required=True,
    default=None,
    help="Asset repository (eg. /content/assets/)",
)
@click.option("--predictions-path", required=True, default=None, help="csv file with predictions")
@LabelErrorOptions.cv_folds
@click.option(
    "--focus-class",
    default=None,
    callback=lambda _, __, x: x.split(",") if x else None,
    help="Only display selected class in Mapper graph",
)
@PredictOptions.from_model
@PredictOptions.from_project
def main(
    api_endpoint: str,
    api_key: str,
    project_id: str,
    clear_dataset_cache: bool,
    target_job: List[str],
    model_framework: ModelFrameworkT,
    model_name: ModelNameT,
    model_repository: ModelRepositoryT,
    asset_status_in: Optional[List[AssetStatusT]],
    label_merge_strategy: LabelMergeStrategyT,
    max_assets: int,
    assets_repository: str,
    predictions_path: Optional[str],
    batch_size: int,
    epochs: int,
    cv_folds: int,
    focus_class: Optional[List[str]],
    from_model: Optional[ModelFrameworkT],
    from_project: Optional[str],
):
    """
    Main method for creating mapper
    """

    kili = Kili(api_key=api_key, api_endpoint=api_endpoint)
    input_type, jobs, _ = get_project(kili, project_id)

    if max_assets and max_assets < 10:
        raise ValueError("max_assets should be greater than 10")

    for job_name, job in jobs.items():
        if target_job and job_name not in target_job:
            continue

        kili_print(f"Create Mapper for job: {job_name}")

        content_input = job.get("content", {}).get("input")
        ml_task = job.get("mlTask")
        tools = job.get("tools")
        if content_input == "radio" and ml_task == "CLASSIFICATION" and input_type == "IMAGE":
            # Get assets
            assets = get_assets(
                kili,
                project_id,
                status_in=asset_status_in,
                max_assets=max_assets,
            )
            labeled_assets = []
            labels = []
            for asset in assets:
                label = get_label(asset, label_merge_strategy)
                if (label is None) or (job_name not in label["jsonResponse"]):
                    asset_id = asset["id"]
                    warnings.warn(f"${asset_id}: No annotation for job ${job_name}")
                else:
                    labeled_assets.append(asset)
                    labels.append(label["jsonResponse"][job_name]["categories"][0]["name"])

            if predictions_path is None:

                image_classification_model = PyTorchVisionImageClassificationModel(
                    model_repository=model_repository,
                    model_name=model_name,
                    job=job,
                    model_framework=model_framework,
                    job_name=job_name,
                    project_id=project_id,
                )

                training_loss = image_classification_model.train(
                    assets=labeled_assets,
                    label_merge_strategy=label_merge_strategy,
                    batch_size=batch_size,
                    epochs=epochs,
                    clear_dataset_cache=clear_dataset_cache,
                    disable_wandb=True,
                    api_key=api_key,
                    verbose=4,
                )

                training_losses = [job_name, training_loss]
                print(tabulate(training_losses, headers=["job_name", "training_loss"]))

                job_predictions = predict_one_job(
                    api_key=api_key,
                    api_endpoint=api_endpoint,
                    project_id=project_id,
                    from_model=from_model,
                    verbose=4,
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

                predictions = job_predictions.predictions_probability
            else:
                predictions = list(pd.read_csv(predictions_path))

            mapper_image_classification = MapperClassification(
                api_key=api_key,
                input_type=input_type,
                assets=assets,
                labels=labels,
                job=job,
                job_name=job_name,
                assets_repository=assets_repository,
                predictions=predictions,
                focus_class=focus_class,
            )

            _ = mapper_image_classification.create_mapper(cv_folds)

        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
