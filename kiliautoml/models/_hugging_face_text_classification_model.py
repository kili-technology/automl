# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
import json
import os
from typing import Any, Dict, List, Optional

import datasets
import evaluate  # type: ignore
import nltk
import numpy as np
from tqdm.autonotebook import tqdm
from transformers import Trainer

from kiliautoml.mixins._hugging_face_mixin import HuggingFaceMixin
from kiliautoml.mixins._kili_text_project_mixin import KiliTextProjectMixin
from kiliautoml.models._base_model import BaseModel
from kiliautoml.utils.helpers import (
    JobPredictions,
    categories_from_job,
    ensure_dir,
    kili_print,
)
from kiliautoml.utils.path import Path, PathHF
from kiliautoml.utils.type import (
    AdditionalTrainingArgsT,
    AssetT,
    JobNameT,
    JobT,
    JsonResponseClassification,
    MLTaskT,
    ModelFrameworkT,
    ModelMetricT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
)


class HuggingFaceTextClassificationModel(BaseModel, HuggingFaceMixin, KiliTextProjectMixin):

    ml_task: MLTaskT = "CLASSIFICATION"
    model_repository: ModelRepositoryT = "huggingface"

    advised_model_names: List[ModelNameT] = [
        "bert-base-multilingual-cased",
        "distilbert-base-cased",
        "distilbert-base-uncased",
    ]

    def __init__(
        self,
        *,
        project_id: ProjectIdT,
        api_key: str,
        api_endpoint: str,
        job_name: JobNameT,
        job: JobT,
        model_name: ModelNameT = "bert-base-multilingual-cased",
        model_framework: ModelFrameworkT = "pytorch",
    ) -> None:
        KiliTextProjectMixin.__init__(self, project_id, api_key, api_endpoint)
        BaseModel.__init__(
            self, job=job, job_name=job_name, model_name=model_name, model_framework=model_framework
        )

    def train(
        self,
        *,
        assets: List[AssetT],
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool = False,
        disable_wandb: bool = False,
        verbose: int,
        additional_train_args_hg: AdditionalTrainingArgsT,
    ):
        _ = verbose

        nltk.download("punkt")

        model_repository_dir = Path.model_repository_dir(
            self.project_id, self.job_name, self.model_repository
        )

        model_name: ModelNameT = self.model_name  # type: ignore

        kili_print(self.job_name)
        path_dataset = os.path.join(PathHF.dataset_dir(model_repository_dir), "data.json")
        kili_print(f"Downloading data to {path_dataset}")
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
            model_name, self.model_framework, job_categories, self.ml_task
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["test"]  # type: ignore
        path_model = PathHF.append_model_folder(model_repository_dir, self.model_framework)

        training_arguments = self._get_training_args(
            path_model,
            model_name,
            disable_wandb=disable_wandb,
            epochs=epochs,
            batch_size=batch_size,
            additional_train_args_hg=additional_train_args_hg,
        )

        trainer = Trainer(
            model=model,
            args=training_arguments,
            tokenizer=tokenizer,
            train_dataset=train_dataset,  # type: ignore
            eval_dataset=eval_dataset,  # type: ignore
            compute_metrics=self.compute_metrics,  # type: ignore
        )
        trainer.train()
        model_evaluation = self.model_evaluation(trainer, job_categories)

        kili_print(f"Saving model to {path_model}")
        trainer.save_model(ensure_dir(path_model))
        return dict(sorted(model_evaluation.items()))

    def predict(
        self,
        *,
        assets: List[AssetT],
        model_path: Optional[str],
        from_project: Optional[str],
        batch_size: int,
        verbose: int,
        clear_dataset_cache: bool,
    ) -> JobPredictions:
        print("Warning, this model does not support custom batch_size ", batch_size)
        _ = clear_dataset_cache

        model_path_res, _, self.model_framework = self._extract_model_info(
            self.job_name, self.project_id, model_path, from_project
        )

        predictions = []
        proba_assets = []

        tokenizer, model = self._get_tokenizer_and_model(
            self.model_framework, model_path_res, self.ml_task
        )

        for asset in assets:
            text = self._get_text_from(asset.content)

            predictions_asset = self._compute_asset_classification(
                self.model_framework, tokenizer, model, text
            )

            predictions.append({self.job_name: predictions_asset})
            proba_assets.append(predictions_asset["categories"][0]["confidence"])

            if verbose:
                print("----------")
                print(text)
                print(predictions_asset)

        # Warning: the granularity of proba_assets is the whole document
        job_predictions = JobPredictions(
            job_name=self.job_name,
            external_id_array=[a.externalId for a in assets],
            json_response_array=predictions,
            model_name_array=["Kili AutoML"] * len(assets),
            predictions_probability=proba_assets,
        )

        return job_predictions

    def _write_dataset(self, assets, job_name, path_dataset, job_categories):
        with open(ensure_dir(path_dataset), "w") as handler:
            for asset in tqdm(assets, desc="Downloading content"):
                label_category = asset.labels[0]["jsonResponse"][job_name]["categories"][0]["name"]
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
        model_framework, tokenizer, model, asset
    ) -> JsonResponseClassification:
        # imposed by the model
        asset = asset[: model.config.max_position_embeddings]

        if model_framework == "pytorch":
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

    def compute_metrics(self, eval_pred):
        metrics = ["accuracy", "precision", "recall", "f1"]
        metric = {}
        for met in metrics:
            if met == "accuracy":
                metric[met] = datasets.load_metric(met)
            else:
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

    def model_evaluation(self, trainer, job_categories):
        train_metrics = trainer.evaluate(trainer.train_dataset)
        val_metrics = trainer.evaluate(trainer.eval_dataset)
        model_evaluation: Dict[str, Any] = {}

        if len(train_metrics["eval_precision"]["by_category"]) == len(job_categories):
            for i, label in enumerate(job_categories):
                train_label_metrics = {}
                for metric in ["precision", "recall", "f1"]:
                    train_label_metrics[metric] = train_metrics["eval_" + metric]["by_category"][i]
                model_evaluation["train_" + label] = train_label_metrics
        if len(val_metrics["eval_precision"]["by_category"]) == len(job_categories):
            for i, label in enumerate(job_categories):
                val_label_metrics = {}
                for metric in ["precision", "recall", "f1"]:
                    val_label_metrics[metric] = val_metrics["eval_" + metric]["by_category"][i]
                model_evaluation["val_" + label] = val_label_metrics
        model_evaluation["train__overall"] = {
            "loss": train_metrics["eval_loss"],
            "accuracy": train_metrics["eval_accuracy"]["overall"],
            "precision": train_metrics["eval_precision"]["overall"],
            "recall": train_metrics["eval_recall"]["overall"],
            "f1": train_metrics["eval_f1"]["overall"],
        }
        model_evaluation["val__overall"] = {
            "loss": val_metrics["eval_loss"],
            "accuracy": val_metrics["eval_accuracy"]["overall"],
            "precision": val_metrics["eval_precision"]["overall"],
            "recall": val_metrics["eval_recall"]["overall"],
            "f1": val_metrics["eval_f1"]["overall"],
        }
        return model_evaluation

    def find_errors(
        self,
        *,
        assets: List[AssetT],
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: int = 0,
        clear_dataset_cache: bool = False,
    ):
        raise NotImplementedError("This model does not support find_errors yet")
