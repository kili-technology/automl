# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
import json
import os
from typing import Dict, List
from warnings import warn

import datasets

import requests
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TFAutoModelForSequenceClassification,
    TFAutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)

from utils.constants import ModelFramework
from utils.helpers import ensure_dir, kili_print, categories_from_job
from utils.huggingface.converters import kili_assets_to_hf_ner_dataset
from utils.path_manager import (
    get_huggingface_train_path,
    get_training_arguments_huggingface,
    get_path_model_huggingface,
)


def huggingface_train_ner(
    api_key: str,
    assets: List[Dict],
    job: Dict,
    job_name: str,
    model_framework: str,
    model_name: str,
    path: str,
    clear_dataset_cache: bool,
) -> float:
    """
    Sources:
     - https://huggingface.co/transformers/v2.4.0/examples.html#named-entity-recognition
     - https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py # noqa
     - https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=okwWVFwfYKy1  # noqa
    """
    kili_print(f"Job Name: {job_name}")
    kili_print(f"Base model: {model_name}")
    path_dataset = get_huggingface_train_path(path)

    label_list = kili_assets_to_hf_ner_dataset(
        api_key, job, job_name, path_dataset, assets, clear_dataset_cache
    )

    raw_datasets = datasets.load_dataset(
        "json",
        data_files=path_dataset,
        features=datasets.features.features.Features(
            {
                "ner_tags": datasets.Sequence(feature=datasets.ClassLabel(names=label_list)),
                "tokens": datasets.Sequence(feature=datasets.Value(dtype="string")),
            }
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_all_tokens = True

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

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
    path_model = get_path_model_huggingface(path, model_framework)
    if model_framework == ModelFramework.PyTorch:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(label_list), id2label=dict(enumerate(label_list))
        )
    elif model_framework == ModelFramework.Tensorflow:
        model = TFAutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label_list),
            from_pt=True,
            id2label=dict(enumerate(label_list)),
        )
    else:
        raise NotImplementedError
    training_args = TrainingArguments(get_training_arguments_huggingface(path_model))
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,  # type:ignore
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # type:ignore
    )
    output = trainer.train()
    kili_print(f"Saving model to {path_model}")
    trainer.save_model(ensure_dir(path_model))
    return output.training_loss


def huggingface_train_text_classification_single(
    api_key: str,
    assets: List[Dict],
    job: Dict,
    job_name: str,
    model_framework: str,
    model_name: str,
    path: str,
    clear_dataset_cache: bool,
) -> float:
    """
    Source: https://huggingface.co/docs/transformers/training
    """
    kili_print(job_name)
    path_dataset = get_huggingface_train_path(path)
    kili_print(f"Downloading data to {path_dataset}")
    if os.path.exists(path_dataset) and clear_dataset_cache:
        os.remove(path_dataset)
    job_categories = categories_from_job(job)
    if not os.path.exists(path_dataset):
        with open(ensure_dir(path_dataset), "w") as handler:
            for asset in assets:
                response = requests.get(
                    asset["content"],
                    headers={
                        "Authorization": f"X-API-Key: {api_key}",
                    },
                )
                if job_name in asset["labels"][0]["jsonResponse"]:
                    label_category = asset["labels"][0]["jsonResponse"][job_name]["categories"][0][
                        "name"
                    ]
                    handler.write(
                        json.dumps(
                            {
                                "text": response.text,
                                "label": job_categories.index(label_category),
                            }
                        )
                        + "\n"
                    )
                else:
                    asset_id = asset["id"]
                    warn(f"Asset {asset_id} does not have {job_name} annotation")
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    train_dataset: datasets.Dataset = tokenized_datasets["train"]  # type:ignore
    path_model = get_path_model_huggingface(path, model_framework)
    if model_framework == ModelFramework.PyTorch:

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(job_categories), id2label=dict(enumerate(job_categories))
        )
    elif model_framework == ModelFramework.Tensorflow:
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(job_categories),
            from_pt=True,
            id2label=dict(enumerate(job_categories)),
        )
    else:
        raise NotImplementedError
    training_args = TrainingArguments(get_training_arguments_huggingface(path_model))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type:ignore
        tokenizer=tokenizer,
    )
    output = trainer.train()
    kili_print(f"Saving model to {path_model}")
    trainer.save_model(ensure_dir(path_model))
    return output.training_loss
