import os

import click
from typing_extensions import get_args

from kiliautoml.utils.constants import ModelFrameworkT, ModelNameT


class Options:

    project_id = click.option("--project-id", required=True, help="Kili project ID")

    api_endpoint = click.option(
        "--api-endpoint",
        default="https://cloud.kili-technology.com/api/label/v2/graphql",
        help="Kili Endpoint. Usefull to access staging environment.",
    )

    api_key = click.option(
        "--api-key", default=os.environ.get("KILI_API_KEY", ""), help="Kili API Key"
    )

    model_framework = click.option(
        "--model-framework",
        default="pytorch",
        help="Model framework (eg. pytorch, tensorflow)",
        type=click.Choice(get_args(ModelFrameworkT)),
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

    asset_status_in = click.option(
        "--asset-status-in",
        default=["LABELED", "TO_REVIEW", "REVIEWED"],
        callback=lambda _, __, x: x.upper().split(",") if x else [],
        help=(
            "Comma separated (without space) list of Kili asset status to select "
            "among: 'TODO', 'ONGOING', 'LABELED', 'TO_REVIEW', 'REVIEWED'"
            "Example: python train.py --asset-status-in TO_REVIEW,REVIEWED "
        ),
    )

    target_job = click.option(
        "--target-job",
        default=None,
        multiple=True,
        help=(
            "Add a specific target job on which to train on "
            "(multiple can be passed if --target-job <job_name> is repeated) "
            "Example: python train.py --target-job BBOX --target-job CLASSIFICATION"
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


class TrainOptions:
    epochs = click.option(
        "--epochs",
        default=10,
        type=int,
        show_default=True,
        help="Number of epochs to train for",
    )

    json_args = click.option(
        "--json-args",
        default=None,
        type=str,
        help=(
            "Specific parameters to pass to the trainer "
            "(for example Yolov5 train, Hugging Face transformers, ..."
        ),
    )

    disable_wandb = click.option(
        "--disable-wandb",
        default=False,
        is_flag=True,
        help="Tells if wandb is disabled",
    )


class PredictOptions:

    dry_run = click.option(
        "--dry-run",
        default=False,
        is_flag=True,
        help="Runs the predictions but do not save them into the Kili project",
    )
    from_model = click.option(
        "--from-model",
        default=None,
        help="Runs the predictions using a specified model path",
    )
    from_project = click.option(
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
