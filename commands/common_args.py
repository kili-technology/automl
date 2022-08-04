import json
import os
from typing import List

import click
from typing_extensions import get_args

from kiliautoml.utils.type import AssetStatusT, MLBackendT, ModelNameT


class Options:

    project_id = click.option("--project-id", required=True, help="Kili project ID")

    api_endpoint = click.option(
        "--api-endpoint",
        default="https://cloud.kili-technology.com/api/label/v2/graphql",
        help="Kili API endpoint. Can be used to specify 'on premise' environment",
    )

    api_key = click.option(
        "--api-key",
        default=os.environ.get("KILI_API_KEY", ""),
        help="Kili API Key",
    )

    ml_backend = click.option(
        "--ml-backend",
        default="pytorch",
        help="ml-backend (eg. pytorch, tensorflow)",
        type=click.Choice(get_args(MLBackendT)),
    )
    model_name = click.option(
        "--model-name",
        default=None,
        help="Model name (eg. bert-base-cased)",
        type=click.Choice(get_args(ModelNameT)),
    )

    model_repository = click.option(
        "--model-repository", default=None, help="Model repository (eg. huggingface)"
    )

    target_job = click.option(
        "--target-job",
        default=None,
        multiple=True,
        help=(
            "Add a specific target job on which to train on "
            "(multiple can be passed if --target-job <job_name> is repeated) "
            "Example: python train.py --target-job JOB1 --target-job JOB2"
        ),
    )

    ignore_job = click.option(
        "--ignore-job",
        default=None,
        multiple=True,
        help=(
            "Ignore job on which to train on "
            "(multiple can be passed if --ignore-job <job_name> is repeated) "
            "Example: python train.py --ignore-job JOB1 --ignore-job JOB2"
        ),
    )

    max_assets = click.option(
        "--max-assets",
        default=None,
        type=int,
        help="Maximum number of assets to consider",
    )

    clear_dataset_cache = click.option(
        "--clear-dataset-cache",
        default=False,
        is_flag=True,
        help="Tells if the dataset cache must be cleared",
    )

    batch_size = click.option(
        "--batch-size",
        default=8,
        type=int,
    )

    verbose = click.option("--verbose", default=0, type=int, help="Verbose level")

    randomize_assets = click.option(
        "--randomize-assets",
        default=True,
        type=bool,
        help="Whether or not to shuffle Kili assets",
    )

    label_merge_strategy = click.option(
        "--label-merge-strategy",
        default="last",
        help=(
            "Strategy to select the right label when more than one are available"
            "for one asset. AutoML always select the best type of label ('Review' then "
            "'Default'). When there are several labels for the highest priority label type, "
            "the user can specify if the last label is taken or the first one"
        ),
    )


def asset_status_in(default: List[AssetStatusT]):
    default_string = ",".join(default) if default else None
    return click.option(
        "--asset-status-in",
        default=default_string,  # type: ignore
        callback=lambda _, __, x: x.upper().split(",") if x else None,
        help=(
            "Comma separated (without space) list of Kili asset status to select "
            "among: 'TODO', 'ONGOING', 'LABELED', 'TO_REVIEW', 'REVIEWED'"
            "Example: python train.py --asset-status-in TO_REVIEW,REVIEWED "
        ),
    )


class TrainOptions:
    epochs = click.option(
        "--epochs",
        default=50,
        type=int,
        show_default=True,
        help="Number of epochs to train for",
    )

    disable_wandb = click.option(
        "--disable-wandb",
        default=False,
        is_flag=True,
        help="Tells if wandb is disabled",
    )

    asset_status_in = asset_status_in(["LABELED", "TO_REVIEW", "REVIEWED"])

    json_string_hg = '{"logging_strategy": "epoch"}'
    additionalTrainArgsHuggingFace = click.option(
        "--additional-train-args-hg",
        default=json_string_hg,
        callback=lambda _, __, x: json.loads(x),
        help=(
            "args passed to huggingface TrainingArguments constructor. "
            "See https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments"  # noqa
            "Ex:  --additional-train-args-hg " + json_string_hg
        ),
    )

    json_string_yolo = '{"bbox_interval": -1}'
    additionalTrainArgsYolo = click.option(
        "--additional-train-args-yolo",
        default=json_string_yolo,
        callback=lambda _, __, x: json.loads(x),
        help=(
            "Additional args passed to Yolo Training script."
            "See https://github.com/ultralytics/yolov5/blob/master/train.py"
            "Ex:  --additional-train-args-yolo " + json_string_yolo
        ),
    )


class PredictOptions:

    dry_run = click.option(
        "--dry-run",
        default=False,
        is_flag=True,
        help="Runs the predictions but do not save them into the Kili project",
    )
    model_path = click.option(
        "--model-path",
        default=None,
        help="Runs the predictions using a specified model path",
    )
    from_project = click.option(
        "--from-project",
        default=None,
        type=str,
        help=(
            "Use a model trained of a different project to predict on project_id."
            "This argument is ignored if --from-model is used."
        ),
    )

    asset_status_in = asset_status_in(["TODO", "ONGOING"])


class LabelErrorOptions:

    cv_folds = click.option(
        "--cv-folds", default=4, type=int, show_default=True, help="Number of CV folds to use"
    )
    dry_run = click.option(
        "--dry-run",
        default=None,
        is_flag=True,
        help=(
            "Get the labeling errors, save on the hard drive, but do not upload them into the Kili"
            " project"
        ),
    )

    asset_status_in = asset_status_in(["LABELED", "TO_REVIEW", "REVIEWED"])

    erase_error_metadata = click.option(
        "--erase-error-metadata",
        default=None,
        is_flag=True,
        help="Erase annotation errors.",
    )


class PrioritizeOptions:

    diversity_sampling = click.option(
        "--diversity-sampling",
        default=0.3,
        type=float,
        help="Diversity sampling proportion",
    )
    uncertainty_sampling = click.option(
        "--uncertainty-sampling",
        default=0.4,
        type=float,
        help="Uncertainty sampling proportion",
    )

    asset_status_in = asset_status_in(["TODO", "ONGOING"])
