# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
import json
import os
import warnings
from typing import List, Optional

import datasets
import nltk
import numpy as np
from kili.client import Kili
from tqdm.auto import tqdm
from transformers import DataCollatorForTokenClassification, Trainer

from kiliautoml.mixins._hugging_face_mixin import HuggingFaceMixin
from kiliautoml.mixins._kili_text_project_mixin import KiliTextProjectMixin
from kiliautoml.models._base_model import (
    BaseInitArgs,
    KiliBaseModel,
    ModelConditions,
    ModelTrainArgs,
)
from kiliautoml.utils.helpers import categories_from_job, ensure_dir
from kiliautoml.utils.logging import logger
from kiliautoml.utils.path import Path, PathHF
from kiliautoml.utils.type import (
    AdditionalTrainingArgsT,
    AssetsLazyList,
    AssetT,
    CategoriesT,
    CategoryIdT,
    CategoryT,
    JobNameT,
    JobPredictions,
    JobT,
    KiliNerAnnotation,
    MLBackendT,
    ModelNameT,
    ProjectIdT,
)


class HuggingFaceNamedEntityRecognitionModel(KiliBaseModel, HuggingFaceMixin, KiliTextProjectMixin):
    model_conditions = ModelConditions(
        ml_task="NAMED_ENTITIES_RECOGNITION",
        model_repository="huggingface",
        possible_ml_backend=["pytorch", "tensorflow"],
        advised_model_names=[
            ModelNameT("bert-base-cased"),
            ModelNameT("bert-base-multilingual-cased"),
            ModelNameT("distilbert-base-cased"),
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
        clear_dataset_cache: bool,
        disable_wandb: bool,
        additional_train_args_hg: AdditionalTrainingArgsT = {},
        model_train_args: ModelTrainArgs,
    ):
        """
        Sources:
        - https://huggingface.co/transformers/v2.4.0/examples.html#named-entity-recognition
        - https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py # noqa
        - https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=okwWVFwfYKy1  # noqa
        """
        _ = model_train_args
        nltk.download("punkt")

        model_repository_dir = Path.model_repository_dir(
            self.project_id, self.job_name, self.model_repository
        )
        model_name: ModelNameT = self.model_name  # type: ignore
        logger.info(f"JobT Name: {self.job_name}")
        logger.info(f"Base model: {model_name}")
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
        )[
            "train"
        ].train_test_split(  # type: ignore
            test_size=0.1
        )
        tokenizer, model = self._get_tokenizer_and_model_from_name(
            model_name, self.ml_backend, label_list, self.model_conditions.ml_task
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
        eval_dataset = tokenized_datasets["test"]
        path_model = PathHF.append_model_folder(model_repository_dir, self.ml_backend)

        training_arguments = self._get_training_args(
            path_model,
            model_name,
            disable_wandb=disable_wandb,
            epochs=epochs,
            batch_size=batch_size,
            additional_train_args_hg=additional_train_args_hg,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        def compute_metrics(eval_preds):
            metric = datasets.load_metric("seqeval")
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)
            # Remove ignored index (special tokens with label_id -100) and convert to labels
            true_labels = [
                [label_list[label_id] for label_id in label if label_id != -100] for label in labels
            ]
            true_predictions = [
                [label_list[p] for (p, label_id) in zip(prediction, label) if label_id != -100]
                for prediction, label in zip(predictions, labels)
            ]
            all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
            return all_metrics

        trainer = Trainer(
            model=model,
            args=training_arguments,
            data_collator=data_collator,  # type: ignore
            tokenizer=tokenizer,
            train_dataset=train_dataset,  # type: ignore
            eval_dataset=eval_dataset,  # type: ignore
            compute_metrics=compute_metrics,  # type: ignore
        )
        trainer.train()  # type: ignore
        model_evaluation = self.evaluation(trainer)
        logger.info(f"Saving model to {path_model}")
        trainer.save_model(ensure_dir(path_model))  # type: ignore
        return dict(sorted(model_evaluation.items()))

    def predict(
        self,
        *,
        assets: AssetsLazyList,
        model_path: Optional[str],
        from_project: Optional[ProjectIdT],
        batch_size: int,
        clear_dataset_cache: bool,
    ) -> JobPredictions:
        _ = clear_dataset_cache
        warnings.warn("Warning, this method does not support custom batch_size")
        _ = batch_size
        model_path_res, _, self.ml_backend = self._extract_model_info(
            self.job_name,
            self.project_id,
            model_path,
            from_project,
        )

        tokenizer, model = self._get_tokenizer_and_model(
            self.ml_backend, model_path_res, self.model_conditions.ml_task
        )

        predictions = []
        proba_assets = []
        for asset in assets.iter_refreshed_asset(
            kili=Kili(
                api_key=self.api_key,  # TODO: add endpoint
            )
        ):  # TODO: add api_endpoint
            text = self._get_text_from(asset.content)

            offset = 0
            predictions_asset: List[KiliNerAnnotation] = []

            probas_asset = []
            for sentence in nltk.sent_tokenize(text):
                offset_inc = text[offset:].find(sentence)
                if offset_inc == -1:
                    raise Exception(f"Sentence {sentence} not found in text!")
                offset += offset_inc

                predictions_sentence, probas = self._compute_sentence_predictions(
                    self.ml_backend, tokenizer, model, sentence, offset
                )
                probas_asset.append(min(probas))

                predictions_asset.extend(predictions_sentence)
            predictions.append({self.job_name: {"annotations": predictions_asset}})
            proba_assets.append(min(probas_asset))

            if len(predictions_asset):
                for p in predictions_asset:
                    logger.debug(p)
            else:
                logger.debug("No prediction")

        # Warning: the granularity of proba_assets is the whole document
        job_predictions = JobPredictions(
            job_name=self.job_name,
            external_id_array=[a.externalId for a in assets],  # type:ignore
            json_response_array=predictions,
            model_name_array=["Kili AutoML"] * len(assets),
            predictions_probability=proba_assets,
        )

        return job_predictions

    def _kili_assets_to_hf_ner_dataset(
        self,
        job: JobT,
        job_name: JobNameT,
        path_dataset: str,
        assets: AssetsLazyList,
        clear_dataset_cache: bool,
    ) -> List[CategoryIdT]:
        if clear_dataset_cache and os.path.exists(path_dataset):
            logger.info("Dataset cache for this project is being cleared.")
            os.remove(path_dataset)

        job_categories = categories_from_job(job=job)
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

        return [CategoryIdT(cat) for cat in label_list]

    def _write_asset(self, job_name: JobNameT, labels_to_ids, handler, asset: AssetT):
        text = self._get_text_from(asset.content)
        if asset.has_asset_for(job_name):
            annotations = asset.get_annotations_ner(job_name)["annotations"]
        else:
            annotations = []
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
    def _compute_sentence_predictions(
        cls, ml_backend: MLBackendT, tokenizer, model, sentence: str, offset: int
    ):
        # imposed by the model
        sequence = sentence[: model.config.max_position_embeddings]

        if ml_backend == "pytorch":
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
    def _post_process_labels(cls, predicted_labels: List[str], tokens: List[str], null_category):

        post_processed_labels: List[str] = []
        prev_category: Optional[str] = None
        category: Optional[str] = None

        for label, token in zip(predicted_labels, tokens):
            if token in [
                "[CLS]",
                "[SEP]",
                "[UNK]",
            ]:  # special BERT tokens that should ignored at inference time
                post_processed_labels.append(label)
                continue

            if token.startswith(
                "##"
            ):  # hash token annotations should be ignored when aligning tokens and text
                token = token.replace("##", "")

            category = None
            if label != null_category:
                category = label.replace("B-", "").replace("I-", "")

                if len(post_processed_labels):
                    if category == prev_category:
                        new_label = "I-" + category
                    else:
                        new_label = "B-" + category
                else:
                    new_label = label
            else:
                new_label = label
            post_processed_labels.append(new_label)

            prev_category = category

        return post_processed_labels

    @classmethod
    def _compute_kili_annotations(
        cls,
        text: str,
        labels: List[str],
        probas: List[float],
        tokens: List[str],
        null_category: str,
        offset_in_text: int,
    ) -> List[KiliNerAnnotation]:

        offset_in_sentence: int = 0
        kili_annotations: List[KiliNerAnnotation] = []

        for label, proba, token in zip(labels, probas, tokens):
            if token in [
                "[CLS]",
                "[SEP]",
                "[UNK]",
            ]:  # special BERT tokens that should ignored at inference time
                continue
            if token.startswith(
                "##"
            ):  # hash token annotations should be ignored when aligning tokens and text
                token = token.replace("##", "")
            text_remaining = text[offset_in_sentence:]
            ind_in_remaining_text = text_remaining.lower().find(token.lower())
            if ind_in_remaining_text == -1:
                raise Exception(f"token '{token}' not found in text '{text_remaining}'")

            content = token
            str_between_tokens = text_remaining[:ind_in_remaining_text]

            if label != null_category:

                categories: CategoriesT = [
                    CategoryT(name=CategoryIdT(label[2:]), confidence=int(proba * 100))
                ]
                ann: KiliNerAnnotation = {
                    "beginOffset": offset_in_text + offset_in_sentence + ind_in_remaining_text,
                    "content": content,
                    "endOffset": offset_in_text
                    + offset_in_sentence
                    + ind_in_remaining_text
                    + len(content),
                    "categories": categories,
                }

                if label.startswith("I-"):
                    # merge with previous if continuation label
                    kili_annotations[-1]["endOffset"] = ann["endOffset"]
                    kili_annotations[-1]["content"] += str_between_tokens + ann["content"]

                else:
                    kili_annotations.append(ann)

            offset_in_sentence += ind_in_remaining_text + len(token)

        return kili_annotations

    @classmethod
    def _predicted_tokens_to_kili_annotations(
        cls,
        text: str,
        predicted_labels: List[str],
        predicted_probas: List[float],
        tokens: List[str],
        null_category: str,
        offset_in_text: int,
    ) -> List[KiliNerAnnotation]:
        """
        Format token predictions into a the kili format.
        :param: text:
        """
        pp_labels = cls._post_process_labels(predicted_labels, tokens, null_category)

        return cls._compute_kili_annotations(
            text, pp_labels, predicted_probas, tokens, null_category, offset_in_text
        )

    def evaluation(self, trainer):
        train_metrics = trainer.evaluate(trainer.train_dataset)
        val_metrics = trainer.evaluate(trainer.eval_dataset)
        model_evaluation = {}
        nb_train_ent = 0
        nb_val_ent = 0
        for label in categories_from_job(self.job):
            if "eval_" + label in train_metrics:
                model_evaluation["train_" + label] = train_metrics["eval_" + label]
                nb_train_ent += train_metrics["eval_" + label]["number"]  # type: ignore
            else:
                model_evaluation["train_" + label] = {"number": 0}
            if "eval_" + label in val_metrics:
                model_evaluation["val_" + label] = val_metrics["eval_" + label]
                nb_val_ent += val_metrics["eval_" + label]["number"]  # type: ignore
            else:
                model_evaluation["val_" + label] = {"number": 0}

        model_evaluation["train__overall"] = {
            "loss": train_metrics["eval_loss"],
            "precision": train_metrics["eval_overall_precision"],
            "recall": train_metrics["eval_overall_recall"],
            "f1": train_metrics["eval_overall_f1"],
            "number": nb_train_ent,
        }
        model_evaluation["val__overall"] = {
            "loss": val_metrics["eval_loss"],
            "precision": val_metrics["eval_overall_precision"],
            "recall": val_metrics["eval_overall_recall"],
            "f1": val_metrics["eval_overall_f1"],
            "number": nb_val_ent,
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
