from typing import Dict, List, Union
import requests

from nltk import sent_tokenize
import numpy as np
from transformers import AutoTokenizer
from transformers import (
    AutoModelForTokenClassification,
    TFAutoModelForTokenClassification,
)
from utils.constants import ModelFramework
from utils.huggingface.converters import predicted_tokens_to_kili_annotations


def huggingface_predict_ner(
    api_key: str,
    assets: Union[List[Dict], List[str]],
    model_framework: str,
    model_path: str,
    job_name: str,
    verbose: int = 0,
):

    if model_framework == ModelFramework.PyTorch:
        tokenizer = AutoTokenizer.from_pretrained(model_path, from_pt=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    elif model_framework == ModelFramework.Tensorflow:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModelForTokenClassification.from_pretrained(model_path)
    else:
        raise NotImplementedError(
            f"Predictions with model framework {model_framework} not implemented"
        )

    predictions = []

    for asset in assets:
        response = requests.get(
            asset["content"],
            headers={
                "Authorization": f"X-API-Key: {api_key}",
            },
        )

        offset = 0
        predictions_asset = []

        # sum_len_sentences = sum(len(sentence) for sentence in sent_tokenize(response.text))
        # num_sentences = len(list(sent_tokenize(response.text)))
        # len_text = len(response.text)
        # print(f"sum sentences:{sum_len_sentences}")
        # print(f"len_text:{len_text}")
        # print(f"num_sentences:{num_sentences}")
        # return 0
        text = response.text
        for sentence in sent_tokenize(text):
            offset_inc = text[offset:].find(sentence)
            if offset_inc == -1:
                raise Exception(f"Sentence {sentence} not found in text!")
            offset += offset_inc

            predictions_sentence = compute_sentence_predictions(
                model_framework, tokenizer, model, sentence, offset
            )

            predictions_asset.extend(predictions_sentence)

        predictions.append({job_name: {"annotations": predictions_asset}})

        if verbose:
            print(sentence)
            for p in predictions_asset:
                print(p)

    return predictions


def compute_sentence_predictions(model_framework, tokenizer, model, sentence, offset):
    sequence = sentence[: model.config.max_position_embeddings]  # imposed by the model

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
    )  # by convention we consider that the null category is the first one in the label list, hence model.config.id2label[0]

    return predictions_sentence
