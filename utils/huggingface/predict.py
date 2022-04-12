from typing import Dict, List, Union
import requests

from nltk import sent_tokenize
import numpy as np
from transformers import AutoTokenizer  # type: ignore
from transformers import (
    AutoModelForTokenClassification,  # type: ignore
    TFAutoModelForTokenClassification,  # type: ignore
)
from utils.constants import ModelFramework
from utils.helpers import JobPredictions
from utils.huggingface.converters import predicted_tokens_to_kili_annotations


def huggingface_predict_ner(
    api_key: str,
    assets: Union[List[Dict], List[str]],
    model_framework: str,
    model_path: str,
    job_name: str,
    verbose: int = 0,
) -> JobPredictions:

    if model_framework == ModelFramework.PyTorch:
        tokenizer = AutoTokenizer.from_pretrained(model_path, from_pt=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path)  # type: ignore
    elif model_framework == ModelFramework.Tensorflow:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModelForTokenClassification.from_pretrained(model_path)  # type: ignore
    else:
        raise NotImplementedError(
            f"Predictions with model framework {model_framework} not implemented"
        )

    predictions = []

    proba_assets = []
    for asset in assets:
        response = requests.get(
            asset["content"],  # type: ignore
            headers={
                "Authorization": f"X-API-Key: {api_key}",
            },
        )

        offset = 0
        predictions_asset = []

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

            predictions_asset.extend(predictions_sentence)

        predictions.append({job_name: {"annotations": predictions_asset}})
        proba_assets.append(min(probas_asset))

        if verbose:
            for p in predictions_asset:
                print(p)

    # Warning: the granularity of proba_assets is the whole document
    job_predictions = JobPredictions(
        job_name=job_name,
        external_id_array=[a["externalId"] for a in assets],  # type: ignore
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
