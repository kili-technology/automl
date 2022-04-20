# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
from typing import Dict, List, Union
import requests

from nltk import sent_tokenize
import numpy as np
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TFAutoModelForSequenceClassification,
    TFAutoModelForTokenClassification,
)
from utils.constants import ModelFramework
from utils.helpers import JobPredictions
from utils.huggingface.converters import predicted_tokens_to_kili_annotations


def get_tokenizer_and_model(model_framework, model_path, model_type):
    if model_framework == ModelFramework.PyTorch:
        tokenizer = AutoTokenizer.from_pretrained(model_path, from_pt=True)
        if model_type == "ner":
            model = AutoModelForTokenClassification.from_pretrained(model_path)
        elif model_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            raise ValueError("unknown model type")
    elif model_framework == ModelFramework.Tensorflow:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if model_type == "ner":
            model = TFAutoModelForTokenClassification.from_pretrained(model_path)
        elif model_type == "classfication":
            model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            raise ValueError("unknown model type")
    else:
        raise NotImplementedError
    return tokenizer, model


def huggingface_predict_ner(
    api_key: str,
    assets: Union[List[Dict], List[str]],
    model_framework: str,
    model_path: str,
    job_name: str,
    verbose: int = 0,
) -> JobPredictions:

    tokenizer, model = get_tokenizer_and_model(model_framework, model_path, "ner")

    predictions = []

    proba_assets = []
    for asset in assets:
        response = requests.get(
            asset["content"],  # type:ignore
            headers={
                "Authorization": f"X-API-Key: {api_key}",
            },
        )

        offset = 0
        predictions_asset: List[dict] = []

        text = response.text

        probas_asset = []
        for sentence in sent_tokenize(text):
            offset_inc = text[offset:].find(sentence)
            if offset_inc == -1:
                raise Exception(f"Sentence {sentence} not found in text!")
            offset += offset_inc

            predictions_sentence, probas = compute_sentence_predictions(
                model_framework, tokenizer, model, sentence, offset
            )
            probas_asset.append(min(probas))

            predictions_asset.extend(predictions_sentence)  # type:ignore

        predictions.append({job_name: {"annotations": predictions_asset}})
        proba_assets.append(min(probas_asset))

        if verbose:
            if len(predictions_asset):
                for p in predictions_asset:
                    print(p)
            else:
                print("No prediction")

    # Warning: the granularity of proba_assets is the whole document
    job_predictions = JobPredictions(
        job_name=job_name,
        external_id_array=[a["externalId"] for a in assets],  # type:ignore
        json_response_array=predictions,
        model_name_array=["Kili AutoML"] * len(assets),
        predictions_probability=proba_assets,
    )

    return job_predictions


def huggingface_predict_classification(
    api_key: str,
    assets: List[Dict],
    model_framework: str,
    model_path: str,
    job_name: str,
    verbose: int = 0,
) -> JobPredictions:
    if model_framework == ModelFramework.PyTorch:
        tokenizer = AutoTokenizer.from_pretrained(model_path, from_pt=True, truncation=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    elif model_framework == ModelFramework.Tensorflow:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModelForTokenClassification.from_pretrained(model_path)
    else:
        raise NotImplementedError(
            f"Predictions with model framework {model_framework} not implemented"
        )

    predictions = []
    proba_assets = []

    for asset in assets:
        tokenizer, model = get_tokenizer_and_model(model_framework, model_path, "classification")

        response = requests.get(
            asset["content"],
            headers={
                "Authorization": f"X-API-Key: {api_key}",
            },
        )

        text = response.text

        predictions_asset = compute_asset_classification(model_framework, tokenizer, model, text)

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


def compute_sentence_predictions(model_framework, tokenizer, model, sentence, offset):
    # imposed by the model
    sequence = sentence[: model.config.max_position_embeddings]

    if model_framework == ModelFramework.PyTorch:
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

    predictions_sentence = predicted_tokens_to_kili_annotations(
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


def compute_asset_classification(model_framework, tokenizer, model, asset):
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
