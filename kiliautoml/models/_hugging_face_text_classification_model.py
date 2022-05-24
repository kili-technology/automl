# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
import json
import os
from typing import List, Optional
from warnings import warn

import datasets
import nltk
import numpy as np
from transformers import Trainer
from typing_extensions import Literal

from kiliautoml.mixins._hugging_face_mixin import HuggingFaceMixin
from kiliautoml.mixins._kili_text_project_mixin import KiliTextProjectMixin
from kiliautoml.models._base_model import BaseModel
from kiliautoml.utils.constants import (
    HOME,
    MLTaskT,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
)
from kiliautoml.utils.helpers import (
    JobPredictions,
    categories_from_job,
    ensure_dir,
    kili_print,
)
from kiliautoml.utils.path import Path, PathHF
from kiliautoml.utils.type import AssetT, JobT, TrainingArgsT


class HuggingFaceTextClassificationModel(BaseModel, HuggingFaceMixin, KiliTextProjectMixin):

    ml_task: MLTaskT = "CLASSIFICATION"
    model_repository: ModelRepositoryT = "huggingface"

    def __init__(
        self,
        project_id: str,
        api_key: str,
        api_endpoint: str,
        job_name: str,
        job: JobT,
        model_name: Literal[
            "bert-base-multilingual-cased", "distilbert-base-cased"
        ] = "bert-base-multilingual-cased",
        model_framework: ModelFrameworkT = "pytorch",
    ) -> None:
        KiliTextProjectMixin.__init__(self, project_id, api_key, api_endpoint)
        BaseModel.__init__(
            self, job=job, job_name=job_name, model_name=model_name, model_framework=model_framework
        )

    def train(
        self,
        assets: List[AssetT],
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool = False,
        disable_wandb: bool = False,
        training_args: TrainingArgsT = {},
    ) -> float:

        nltk.download("punkt")

        model_repository_dir = Path.model_repository_dir(
            HOME, self.project_id, self.job_name, self.model_repository
        )

        model_name: ModelNameT = self.model_name  # type: ignore
        training_args = training_args or {}

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
        )
        tokenizer, model = self._get_tokenizer_and_model_from_name(
            model_name, self.model_framework, job_categories, self.ml_task
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        train_dataset = tokenized_datasets["train"]  # type: ignore

        path_model = PathHF.append_model_folder(model_repository_dir, self.model_framework)

        training_arguments = self._get_training_args(
            path_model,
            model_name,
            disable_wandb=disable_wandb,
            epochs=epochs,
            batch_size=batch_size,
            additional_args=training_args,
        )

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,  # type: ignore
            tokenizer=tokenizer,
        )
        output = trainer.train()
        kili_print(f"Saving model to {path_model}")
        trainer.save_model(ensure_dir(path_model))
        return output.training_loss

    def predict(
        self,
        assets: List[AssetT],
        model_path: Optional[str],
        from_project: Optional[str],
        batch_size: int,
        verbose: int,
        clear_dataset_cache: bool,
    ) -> JobPredictions:
        print("Warning, this model does not support custom batch_size ", batch_size)

        model_path_res, _, self.model_framework = self._extract_model_info(
            self.job_name, self.project_id, model_path, from_project
        )

        predictions = []
        proba_assets = []

        tokenizer, model = self._get_tokenizer_and_model(
            self.model_framework, model_path_res, self.ml_task
        )

        for asset in assets:
            text = self._get_text_from(asset["content"])

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
            external_id_array=[a["externalId"] for a in assets],
            json_response_array=predictions,
            model_name_array=["Kili AutoML"] * len(assets),
            predictions_probability=proba_assets,
        )

        return job_predictions

    def _write_dataset(self, assets, job_name, path_dataset, job_categories):
        with open(ensure_dir(path_dataset), "w") as handler:
            for asset in assets:
                if job_name in asset["labels"][0]["jsonResponse"]:
                    label_category = asset["labels"][0]["jsonResponse"][job_name]["categories"][0][
                        "name"
                    ]
                    handler.write(
                        json.dumps(
                            {
                                "text": self._get_text_from(asset["content"]),
                                "label": job_categories.index(label_category),
                            }
                        )
                        + "\n"
                    )
                else:
                    asset_id = asset["id"]
                    warn(f"Asset {asset_id} does not have {job_name} annotation")

    @staticmethod
    def _compute_asset_classification(model_framework, tokenizer, model, asset):
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

    def find_errors(
        self,
        assets: List[AssetT],
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: int = 0,
        clear_dataset_cache: bool = False,
    ):
        raise NotImplementedError("This model does not support find_errors yet")
