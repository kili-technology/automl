from datetime import datetime
import json
import os
from typing import List, Dict, Optional
from warnings import warn

import datasets

from transformers import (
    Trainer,
    TrainingArguments,
)
import numpy as np

from kiliautoml.mixins._hugging_face_mixin import HuggingFaceMixin
from kiliautoml.mixins._kili_text_project_mixin import KiliTextProjectMixin
from kiliautoml.models._base_model import BaseModel
from kiliautoml.utils.constants import ModelFramework, ModelName, MLTask, HOME
from kiliautoml.utils.helpers import (
    set_default,
    build_model_repository_path,
    kili_print,
    categories_from_job,
    ensure_dir,
    JobPredictions,
)


class HuggingFaceTextClassificationModel(BaseModel, HuggingFaceMixin, KiliTextProjectMixin):

    ml_task = MLTask.Classification

    def __init__(self, project_id: str, api_key: str, api_endpoint: str) -> None:
        KiliTextProjectMixin.__init__(self, project_id, api_key, api_endpoint)
        BaseModel.__init__(self)

    def train(
        self,
        assets: List[Dict],
        job: Dict,
        job_name: str,
        model_framework: Optional[ModelFramework],
        model_name: Optional[ModelName],
        clear_dataset_cache: bool = False,
    ) -> float:

        import nltk

        nltk.download("punkt")

        path = build_model_repository_path(HOME, self.project_id, job_name, self.model_repository)

        self.model_framework = set_default(
            model_framework,
            ModelFramework.PyTorch,
            "model_framework",
            [ModelFramework.PyTorch, ModelFramework.Tensorflow],
        )
        model_name = set_default(
            model_name,
            ModelName.BertBaseMultilingualCased,
            "model_name",
            [ModelName.BertBaseMultilingualCased],
        )
        return self._train(
            assets,
            job,
            job_name,
            model_name,
            path,
            clear_dataset_cache,
        )

    def predict(
        self,
        assets: List[Dict],
        model_path: Optional[str],
        job_name: str,
        verbose: int = 0,
    ) -> JobPredictions:

        model_path_res, _, self.model_framework = self._extract_model_info(
            job_name, self.project_id, model_path
        )

        predictions = []
        proba_assets = []

        tokenizer, model = self._get_tokenizer_and_model(
            self.model_framework, model_path_res, self.ml_task
        )

        for asset in assets:
            text = self._get_text_from(asset)

            predictions_asset = self._compute_asset_classification(
                self.model_framework, tokenizer, model, text
            )

            predictions.append({job_name: predictions_asset})
            proba_assets.append(predictions_asset["categories"][0]["confidence"])

            if verbose:
                print("----------")
                print(text)
                print(predictions_asset)

        # Warning: the granularity of proba_assets is the whole document
        job_predictions = JobPredictions(
            job_name=job_name,
            external_id_array=[a["externalId"] for a in assets],
            json_response_array=predictions,
            model_name_array=["Kili AutoML"] * len(assets),
            predictions_probability=proba_assets,
        )

        return job_predictions

    def _train(
        self,
        assets: List[Dict],
        job: Dict,
        job_name: str,
        model_name: str,
        path: str,
        clear_dataset_cache: bool,
    ) -> float:

        kili_print(job_name)
        path_dataset = os.path.join(path, "dataset", "data.json")
        kili_print(f"Downloading data to {path_dataset}")
        if os.path.exists(path_dataset) and clear_dataset_cache:
            os.remove(path_dataset)
        job_categories = categories_from_job(job)
        if not os.path.exists(path_dataset):
            self._write_dataset(assets, job_name, path_dataset, job_categories)
        raw_datasets = datasets.load_dataset(
            "json",
            data_files=path_dataset,
            features=datasets.features.features.Features(
                {
                    "label": datasets.ClassLabel(names=job_categories),
                    "text": datasets.Value(dtype="string"),
                }
            ),
        )
        tokenizer, model = self._get_tokenizer_and_model_from_name(
            model_name, self.model_framework, job_categories, self.ml_task
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)  # type: ignore
        train_dataset = tokenized_datasets["train"]
        path_model = os.path.join(
            path, "model", self.model_framework, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        training_args = TrainingArguments(os.path.join(path_model, "training_args"))
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,  # type: ignore
            tokenizer=tokenizer,
        )
        output = trainer.train()
        kili_print(f"Saving model to {path_model}")
        trainer.save_model(ensure_dir(path_model))
        return output.training_loss

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
                                "text": self._get_text_from(asset),
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

        if model_framework == ModelFramework.PyTorch:
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
