# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
import json
import os
import pathlib
from typing import Any, Dict, Optional

import datasets
import evaluate
import nltk
import numpy as np
import transformers
from kili.client import Kili
from tqdm.autonotebook import tqdm
from transformers import Trainer, TrainingArguments

from commands.common_args import DEFAULT_BATCH_SIZE
from kiliautoml.mixins._hugging_face_mixin import HuggingFaceMixin
from kiliautoml.mixins._kili_text_project_mixin import KiliTextProjectMixin
from kiliautoml.models._base_model import BaseInitArgs, KiliBaseModel, ModelTrainArgs
from kiliautoml.models._hugging_face_model import (
    HuggingFaceModel,
    HuggingFaceModelConditions,
)
from kiliautoml.utils.helpers import categories_from_job, ensure_dir
from kiliautoml.utils.logging import logger
from kiliautoml.utils.path import Path, PathHF
from kiliautoml.utils.type import (
    AssetsLazyList,
    EvalResultsT,
    JobPredictions,
    JsonResponseClassification,
    ModelMetricT,
    ModelNameT,
    ProjectIdT,
)


class HuggingFaceTextClassificationModel(HuggingFaceModel, HuggingFaceMixin, KiliTextProjectMixin):
    model_conditions = HuggingFaceModelConditions(
        ml_task="CLASSIFICATION",
        model_repository="huggingface",
        possible_ml_backend=["pytorch"],
        advised_model_names=[
            ModelNameT("bert-base-multilingual-cased"),
            ModelNameT("distilbert-base-cased"),
            ModelNameT("distilbert-base-uncased"),
        ],
        input_type="TEXT",
        content_input="radio",
        tools=None,
    )

    def __init__(
        self,
        *,
        base_init_args: BaseInitArgs,
    ) -> None:
        KiliTextProjectMixin.__init__(self, base_init_args["api_key"])
        KiliBaseModel.__init__(self, base_init_args)

    def train(
        self,
        *,
        assets: AssetsLazyList,
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool = False,
        disable_wandb: bool = False,
        model_train_args: ModelTrainArgs,
        local_dataset_dir: Optional[pathlib.Path],
    ):
        _ = local_dataset_dir
        nltk.download("punkt")

        model_repository_dir = Path.model_repository_dir(
            self.project_id, self.job_name, self.model_repository
        )

        model_name: ModelNameT = self.model_name  # type: ignore

        path_dataset = os.path.join(PathHF.dataset_dir(model_repository_dir), "data.json")
        logger.info(f"Downloading data to {path_dataset}")
        if not clear_dataset_cache:
            logger.warning(
                "If you are using filter on assets, consider using --clear-dataset-cache, "
                "to make sure that the right assets are used."
            )
        if os.path.exists(path_dataset) and clear_dataset_cache:
            os.remove(path_dataset)
        job_categories = categories_from_job(self.job)
        if not os.path.exists(path_dataset):
            self._write_dataset(assets, self.job_name, path_dataset, job_categories)
        raw_datasets = datasets.load_dataset(  # type: ignore
            "json",
            data_files=path_dataset,
            features=datasets.features.features.Features(  # type: ignore
                {
                    "label": datasets.ClassLabel(names=job_categories),  # type: ignore
                    "text": datasets.Value(dtype="string"),  # type: ignore
                }
            ),
        )[
            "train"
        ].train_test_split(  # type: ignore
            test_size=0.2
        )

        tokenizer, model = self._get_tokenizer_and_model_from_name(
            model_name, self.ml_backend, job_categories, self.model_conditions.ml_task
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["test"]  # type: ignore
        path_model = PathHF.append_model_folder(model_repository_dir, self.ml_backend)

        training_arguments = self._get_training_args(
            path_model,
            model_name,
            disable_wandb=disable_wandb,
            epochs=epochs,
            batch_size=batch_size,
            additional_train_args_hg=model_train_args["additional_train_args_hg"],
        )
        trainer = Trainer(
            model=model,
            args=training_arguments,
            tokenizer=tokenizer,
            train_dataset=train_dataset,  # type: ignore
            eval_dataset=eval_dataset,  # type: ignore
            compute_metrics=self.compute_metrics,  # type: ignore
        )
        trainer.train()  # type: ignore
        model_evaluation = self.model_evaluation(trainer, job_categories)

        logger.info(f"Saving model to {path_model}")
        trainer.save_model(ensure_dir(path_model))  # type: ignore
        return dict(sorted(model_evaluation.items()))

    def eval(
        self,
        *,
        assets: AssetsLazyList,
        batch_size: int,
        clear_dataset_cache: bool = False,
        model_path: Optional[str],
        from_project: Optional[ProjectIdT],
        local_dataset_dir: Optional[pathlib.Path],
    ) -> EvalResultsT:
        _ = local_dataset_dir
        model_repository_dir = Path.model_repository_dir(
            self.project_id, self.job_name, self.model_repository
        )
        path_dataset = os.path.join(PathHF.dataset_dir(model_repository_dir), "data.json")
        job_categories = categories_from_job(self.job)
        if os.path.exists(path_dataset) and clear_dataset_cache:
            os.remove(path_dataset)
        if not os.path.exists(path_dataset):
            self._write_dataset(assets, self.job_name, path_dataset, job_categories)
        dataset = datasets.load_dataset(  # type: ignore
            "json",
            data_files=path_dataset,
            features=datasets.features.features.Features(  # type: ignore
                {
                    "label": datasets.ClassLabel(names=job_categories),  # type: ignore
                    "text": datasets.Value(dtype="string"),  # type: ignore
                }
            ),
        )
        model_path_res, _, self.ml_backend = self._extract_model_info(
            self.job_name, self.project_id, model_path, from_project
        )
        path_model = PathHF.append_model_folder(model_repository_dir, self.ml_backend)
        tokenizer, model = self._get_tokenizer_and_model(
            self.ml_backend, model_path_res, self.model_conditions.ml_task
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        dataset = dataset.map(tokenize_function, batched=True)
        eval_arguments = TrainingArguments(
            PathHF.append_training_args_dir(path_model),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            report_to=["none"],
        )
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=eval_arguments,
            train_dataset=dataset["train"],  # type: ignore
            eval_dataset=dataset["train"],  # type: ignore
            compute_metrics=self.compute_metrics,  # type: ignore
        )
        model_evaluation = self.model_evaluation(trainer, job_categories, eval_only=True)
        return dict(sorted(model_evaluation.items()))

    def predict(
        self,
        *,
        assets: AssetsLazyList,
        model_path: Optional[str],
        from_project: Optional[ProjectIdT],
        batch_size: int,
        clear_dataset_cache: bool,
        local_dataset_dir: Optional[pathlib.Path],
    ) -> JobPredictions:
        if batch_size != DEFAULT_BATCH_SIZE:
            logger.warning("This model does not support custom batch_size ", batch_size)
        _ = clear_dataset_cache, local_dataset_dir

        model_path_res, _, self.ml_backend = self._extract_model_info(
            self.job_name, self.project_id, model_path, from_project
        )

        predictions = []
        proba_assets = []

        tokenizer, model = self._get_tokenizer_and_model(
            self.ml_backend, model_path_res, self.model_conditions.ml_task
        )

        for asset in assets.iter_refreshed_asset(
            project_id=self.project_id,
            kili=Kili(
                self.api_key,
            ),
        ):  # TODO: Add api_endpoint
            text = self._get_text_from(asset.content)

            predictions_asset = self._compute_asset_classification(
                self.ml_backend, tokenizer, model, text
            )

            predictions.append({self.job_name: predictions_asset})
            proba_assets.append(predictions_asset["categories"][0]["confidence"])

            logger.debug("----------")
            logger.debug(text)
            logger.debug(predictions_asset)

        # Warning: the granularity of proba_assets is the whole document
        job_predictions = JobPredictions(
            job_name=self.job_name,
            external_id_array=[a.externalId for a in assets],
            json_response_array=predictions,
            model_name_array=["Kili AutoML"] * len(assets),
            predictions_probability=proba_assets,
        )

        return job_predictions

    def _write_dataset(self, assets: AssetsLazyList, job_name, path_dataset, job_categories):
        with open(ensure_dir(path_dataset), "w") as handler:
            for asset in tqdm(assets, desc="Downloading content"):
                label_category = asset.get_annotations_classification(job_name)["categories"][0][
                    "name"
                ]
                handler.write(
                    json.dumps(
                        {
                            "text": self._get_text_from(asset.content),
                            "label": job_categories.index(label_category),
                        }
                    )
                    + "\n"
                )

    @staticmethod
    def _compute_asset_classification(
        ml_backend, tokenizer, model, asset
    ) -> JsonResponseClassification:
        # imposed by the model
        asset = asset[: model.config.max_position_embeddings]

        if ml_backend == "pytorch":
            tokens = tokenizer(
                asset,
                return_tensors="pt",
                max_length=model.config.max_position_embeddings,
                truncation=True,
            )
        else:
            tokens = tokenizer(
                asset,
                return_tensors="tf",
                max_length=model.config.max_position_embeddings,
                truncation=True,
            )

        output = model(**tokens)
        logits = np.squeeze(output["logits"].detach().numpy())
        probas_all = np.exp(logits) / np.sum(np.exp(logits))
        predicted_id = np.argmax(probas_all).tolist()
        probas = probas_all[predicted_id]
        predicted_label = model.config.id2label[predicted_id]

        return {"categories": [{"name": predicted_label, "confidence": int(probas * 100)}]}

    def compute_metrics(self, eval_pred: transformers.trainer_utils.EvalPrediction):
        metrics = ["accuracy", "precision", "recall", "f1"]
        metric = {}
        for met in metrics:
            metric[met] = evaluate.load(met)
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metric_res = {}
        for met in metrics:
            if met == "accuracy":
                metric_res[met] = ModelMetricT(
                    by_category=None,
                    overall=metric[met].compute(predictions=predictions, references=labels)[met],
                )
            elif met == "f1":
                metric_res[met] = ModelMetricT(
                    by_category=metric[met].compute(
                        predictions=predictions, references=labels, average=None
                    )[met],
                    overall=metric[met].compute(
                        predictions=predictions, references=labels, average="weighted"
                    )[met],
                )
            else:
                metric_res[met] = ModelMetricT(
                    by_category=metric[met].compute(
                        predictions=predictions,
                        references=labels,
                        average=None,
                        zero_division=0,
                    )[met],
                    overall=metric[met].compute(
                        predictions=predictions,
                        references=labels,
                        average="weighted",
                        zero_division=0,
                    )[met],
                )
        return metric_res

    def model_evaluation(self, trainer, job_categories, eval_only=False):
        val_metrics = trainer.evaluate(trainer.eval_dataset)
        metrics_to_consider = [val_metrics]
        dataset_names = ["val"]
        if not eval_only:
            train_metrics = trainer.evaluate(trainer.train_dataset)
            metrics_to_consider.insert(0, train_metrics)
            dataset_names.insert(0, "train")
        model_evaluation: Dict[str, Any] = {}
        for dataset_name, metrics in zip(dataset_names, metrics_to_consider):
            if len(metrics["eval_precision"]["by_category"]) == len(job_categories):
                for i, label in enumerate(job_categories):
                    label_metrics = {}
                    for metric in ["precision", "recall", "f1"]:
                        label_metrics[metric] = metrics["eval_" + metric]["by_category"][i]
                    model_evaluation[f"{dataset_name}_" + label] = label_metrics
            model_evaluation[f"{dataset_name}__overall"] = {
                "loss": metrics["eval_loss"],
                "accuracy": metrics["eval_accuracy"]["overall"],
                "precision": metrics["eval_precision"]["overall"],
                "recall": metrics["eval_recall"]["overall"],
                "f1": metrics["eval_f1"]["overall"],
            }
        return model_evaluation

    def find_errors(
        self,
        *,
        assets: AssetsLazyList,
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool = False,
    ):
        raise NotImplementedError("This model does not support find_errors yet")
