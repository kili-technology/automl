# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
import json
import os
import warnings
from typing import Any, List, Optional

import datasets
import nltk
import numpy as np
from tqdm.auto import tqdm
from transformers import DataCollatorForTokenClassification, Trainer
from typing_extensions import Literal, TypedDict

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
from kiliautoml.utils.helpers import JobPredictions, ensure_dir, kili_print
from kiliautoml.utils.path import Path, PathHF
from kiliautoml.utils.type import AdditionalTrainingArgsT, AssetT, JobT


class KiliNerAnnotations(TypedDict):
    beginOffset: Any
    content: Any
    endOffset: Any
    categories: Any


class HuggingFaceNamedEntityRecognitionModel(BaseModel, HuggingFaceMixin, KiliTextProjectMixin):

    ml_task: MLTaskT = "NAMED_ENTITIES_RECOGNITION"
    model_repository: ModelRepositoryT = "huggingface"

    def __init__(
        self,
        project_id: str,
        api_key: str,
        api_endpoint: str,
        job: JobT,
        job_name: str,
        model_name: Literal[
            "bert-base-multilingual-cased", "distilbert-base-cased"
        ] = "bert-base-multilingual-cased",
        model_framework: ModelFrameworkT = "pytorch",
    ) -> None:
        KiliTextProjectMixin.__init__(self, project_id, api_key, api_endpoint)

        BaseModel.__init__(
            self,
            job=job,
            job_name=job_name,
            model_name=model_name,
            model_framework=model_framework,
        )

    def train(
        self,
        *,
        assets: List[AssetT],
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
        disable_wandb: bool,
        verbose: int,
        additional_train_args_hg: AdditionalTrainingArgsT = {},
    ):
        """
        Sources:
        - https://huggingface.co/transformers/v2.4.0/examples.html#named-entity-recognition
        - https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py # noqa
        - https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=okwWVFwfYKy1  # noqa
        """
        _ = verbose
        nltk.download("punkt")

        model_repository_dir = Path.model_repository_dir(
            HOME, self.project_id, self.job_name, self.model_repository
        )

        model_name: ModelNameT = self.model_name  # type: ignore
        kili_print(f"Job Name: {self.job_name}")
        kili_print(f"Base model: {model_name}")
        path_dataset = os.path.join(PathHF.dataset_dir(model_repository_dir), "data.json")

        label_list = self._kili_assets_to_hf_ner_dataset(
            self.job, self.job_name, path_dataset, assets, clear_dataset_cache
        )

        raw_datasets = datasets.load_dataset(  # type: ignore
            "json",
            data_files=path_dataset,
            features=datasets.features.features.Features(  # type: ignore
                {
                    "ner_tags": datasets.Sequence(  # type: ignore
                        feature=datasets.ClassLabel(names=label_list)  # type: ignore
                    ),  # noqa
                    "tokens": datasets.Sequence(feature=datasets.Value(dtype="string")),  # type: ignore # noqa
                }
            ),
        )

        tokenizer, model = self._get_tokenizer_and_model_from_name(
            model_name, self.model_framework, label_list, self.ml_task
        )

        label_all_tokens = True

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )

            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None.
                    # We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    # For the other tokens in a word,
                    # we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(label[word_idx] if label_all_tokens else -100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

        train_dataset = tokenized_datasets["train"]  # type:  ignore
        path_model = PathHF.append_model_folder(model_repository_dir, self.model_framework)

        training_arguments = self._get_training_args(
            path_model,
            model_name,
            disable_wandb=disable_wandb,
            epochs=epochs,
            batch_size=batch_size,
            additional_train_args_hg=additional_train_args_hg,
        )
        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            model=model,
            args=training_arguments,
            data_collator=data_collator,  # type: ignore
            tokenizer=tokenizer,
            train_dataset=train_dataset,  # type: ignore
        )
        output = trainer.train()
        kili_print(f"Saving model to {path_model}")
        trainer.save_model(ensure_dir(path_model))
        return {"training_loss": output.training_loss}

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
        _ = clear_dataset_cache
        warnings.warn("Warning, this method does not support custom batch_size")
        _ = batch_size
        model_path_res, _, self.model_framework = self._extract_model_info(
            self.job_name,
            self.project_id,
            model_path,
            from_project,
        )

        tokenizer, model = self._get_tokenizer_and_model(
            self.model_framework, model_path_res, self.ml_task
        )

        predictions = []
        proba_assets = []
        for asset in assets:
            text = self._get_text_from(asset["content"])  # type: ignore

            offset = 0
            predictions_asset: List[dict] = []  # type: ignore

            probas_asset = []
            for sentence in nltk.sent_tokenize(text):
                offset_inc = text[offset:].find(sentence)
                if offset_inc == -1:
                    raise Exception(f"Sentence {sentence} not found in text!")
                offset += offset_inc

                predictions_sentence, probas = self._compute_sentence_predictions(
                    self.model_framework, tokenizer, model, sentence, offset
                )
                probas_asset.append(min(probas))

                predictions_asset.extend(predictions_sentence)  # type:ignore

            predictions.append({self.job_name: {"annotations": predictions_asset}})
            proba_assets.append(min(probas_asset))

            if verbose:
                if len(predictions_asset):
                    for p in predictions_asset:
                        kili_print(p)
                else:
                    kili_print("No prediction")

        # Warning: the granularity of proba_assets is the whole document
        job_predictions = JobPredictions(
            job_name=self.job_name,
            external_id_array=[a["externalId"] for a in assets],  # type:ignore
            json_response_array=predictions,
            model_name_array=["Kili AutoML"] * len(assets),
            predictions_probability=proba_assets,
        )

        return job_predictions

    def _kili_assets_to_hf_ner_dataset(
        self,
        job: JobT,
        job_name: str,
        path_dataset: str,
        assets: List[AssetT],
        clear_dataset_cache: bool,
    ):

        if clear_dataset_cache and os.path.exists(path_dataset):
            kili_print("Dataset cache for this project is being cleared.")
            os.remove(path_dataset)

        job_categories = list(job["content"]["categories"].keys())
        label_list = (
            ["O"] + ["B-" + jc for jc in job_categories] + ["I-" + jc for jc in job_categories]
        )

        labels_to_ids = {label: i for i, label in enumerate(label_list)}

        if os.path.exists(path_dataset) and clear_dataset_cache:
            os.remove(path_dataset)
        if not os.path.exists(path_dataset):
            with open(ensure_dir(path_dataset), "w") as handler:
                for asset in tqdm(assets, desc="Converting assets to huggingface dataset"):
                    self._write_asset(job_name, labels_to_ids, handler, asset)

        return label_list

    def _write_asset(self, job_name, labels_to_ids, handler, asset):
        text = self._get_text_from(asset["content"])
        annotations = asset["labels"][0]["jsonResponse"][job_name]["annotations"]
        sentences = nltk.sent_tokenize(text)
        offset = 0
        for sentence_tokens in nltk.TreebankWordTokenizer().span_tokenize_sents(sentences):
            tokens = []
            ner_tags = []
            for start_without_offset, end_without_offset in sentence_tokens:
                start, end = (
                    start_without_offset + offset,
                    end_without_offset + offset,
                )
                token_annotations = [
                    a
                    for a in annotations
                    if a["beginOffset"] <= start and a["beginOffset"] + len(a["content"]) >= end
                ]
                if len(token_annotations) > 0:
                    category = token_annotations[0]["categories"][0]["name"]
                    label = (
                        "B-" + category
                        if token_annotations[0]["beginOffset"] == start
                        else "I-" + category
                    )
                else:
                    label = "O"
                tokens.append(text[start:end])
                ner_tags.append(labels_to_ids[label])
            handler.write(
                json.dumps(
                    {
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
                )
                + "\n"
            )
            offset = offset + sentence_tokens[-1][1] + 1

    @classmethod
    def _compute_sentence_predictions(cls, model_framework, tokenizer, model, sentence, offset):
        # imposed by the model
        sequence = sentence[: model.config.max_position_embeddings]

        if model_framework == "pytorch":
            tokens = tokenizer(
                sequence,
                return_tensors="pt",
                max_length=model.config.max_position_embeddings,
            )
        else:
            tokens = tokenizer(
                sequence,
                return_tensors="tf",
                max_length=model.config.max_position_embeddings,
            )

        output = model(**tokens)

        logits = np.squeeze(output["logits"].detach().numpy())
        probas_all = np.exp(logits) / np.expand_dims(np.sum(np.exp(logits), axis=1), axis=1)
        predicted_ids = np.argmax(probas_all, axis=-1).tolist()
        probas = [probas_all[i, p] for i, p in enumerate(predicted_ids)]
        predicted_labels = [model.config.id2label[p] for p in predicted_ids]

        predictions_sentence = cls._predicted_tokens_to_kili_annotations(
            sequence,
            predicted_labels,
            probas,
            [tokenizer.batch_decode([t])[0] for t in tokens["input_ids"][0]],
            model.config.id2label[0],
            offset,
        )
        # by convention we consider that the null category is the first one in the label list,
        # hence model.config.id2label[0]

        return predictions_sentence, probas

    @classmethod
    def _predicted_tokens_to_kili_annotations(
        cls,
        text: str,
        predicted_label: List[str],
        predicted_proba: List[float],
        tokens: List[str],
        null_category: str,
        offset_in_text: int,
    ) -> List[KiliNerAnnotations]:
        """
        Format token predictions into a the kili format.
        :param: text:
        """

        kili_annotations: List[KiliNerAnnotations] = []
        offset_in_sentence = 0
        for label, proba, token in zip(predicted_label, predicted_proba, tokens):
            if token in [
                "[CLS]",
                "[SEP]",
                "[UNK]",
            ]:  # special BERT tokens that should ignored at inference time
                continue
            if token.startswith(
                "##"
            ):  # number tokens annotation should be ignored when aligning categories
                token = token.replace("##", "")

            text_remaining = text[offset_in_sentence:]
            ind = text_remaining.find(token)
            if ind == -1:
                raise Exception(f"token {token} not found in text {text_remaining}")

            offset_in_sentence += ind

            if label != null_category:
                is_i_tag = label.startswith("I-")
                c_kili = label.replace("B-", "").replace("I-", "")

                ann_ = {
                    "beginOffset": offset_in_text + offset_in_sentence,
                    "content": token,
                    "endOffset": offset_in_text + offset_in_sentence + len(token),
                    "categories": [{"name": c_kili, "confidence": int(proba * 100)}],
                }
                ann = KiliNerAnnotations(
                    beginOffset=ann_["beginOffset"],
                    content=ann_["content"],
                    endOffset=ann_["endOffset"],
                    categories=ann_["categories"],
                )

                if (
                    len(kili_annotations)
                    and ann["categories"][0]["name"]
                    == kili_annotations[-1]["categories"][0]["name"]
                    and (ann["beginOffset"] == kili_annotations[-1]["endOffset"] or is_i_tag)
                ):
                    # merge with previous if same category and contiguous offset and onset:
                    kili_annotations[-1]["endOffset"] = ann["endOffset"]
                    kili_annotations[-1]["content"] += ann["content"]

                else:
                    kili_annotations.append(ann)

            offset_in_sentence += len(token)

        return kili_annotations

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
