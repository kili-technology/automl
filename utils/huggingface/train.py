from datetime import datetime
import json
import os
from pyexpat import features
from typing import Dict, List

import datasets
from nltk import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
import requests
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification, \
    AutoTokenizer, DataCollatorForTokenClassification, \
    TFAutoModelForSequenceClassification, TFAutoModelForTokenClassification, \
    Trainer, TrainingArguments

from utils.constants import ModelFramework
from utils.helpers import ensure_dir, kili_print


def huggingface_train_ner(
        api_key:str, assets:List[Dict], job: Dict, job_name:str,
        model_framework:str, model_name:str, path:str):
    '''
    Sources:
     - https://huggingface.co/transformers/v2.4.0/examples.html#named-entity-recognition
     - https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py
     - https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=okwWVFwfYKy1
    '''
    path_dataset = os.path.join(path, 'dataset', 'data.json')
    kili_print(f'Downloading data to {path_dataset}')
    if os.path.exists(path_dataset):
        os.remove(path_dataset)
    job_categories = list(job['content']['categories'].keys())
    label_list = ['O'] + ['B-' + jc for \
            jc in job_categories] + ['I-' + jc for jc in job_categories]
    labels_to_ids = {
        label: i for i, label in enumerate(label_list)
        }
    with open(ensure_dir(path_dataset), 'w') as handler:
        for asset in assets:
            response = requests.get(asset['content'], headers={
                'Authorization': f'X-API-Key: {api_key}',
            })
            text = response.text
            annotations = asset['labels'][0]['jsonResponse'][job_name]['annotations']
            sentences = sent_tokenize(text)
            offset = 0
            for sentence_tokens in TreebankWordTokenizer().span_tokenize_sents(sentences):
                tokens = []
                ner_tags = []
                for start_without_offset, end_without_offset in sentence_tokens:
                    start, end = start_without_offset + offset, end_without_offset + offset
                    token_annotations = [a for a in annotations \
                        if a['beginOffset'] <= start and a['endOffset'] >= end]
                    if len(token_annotations) > 0:
                        category = token_annotations[0]['categories'][0]['name']
                        label = 'B-' + category if token_annotations[0]['beginOffset'] == start \
                            else 'I-' + category
                    else:
                        label = 'O'
                    tokens.append(text[start:end])
                    ner_tags.append(labels_to_ids[label])
                handler.write(json.dumps({
                    'tokens': tokens,
                    'ner_tags': ner_tags,
                }) + '\n')
                offset = offset + sentence_tokens[-1][1] + 1
    raw_datasets = datasets.load_dataset('json', 
        data_files=path_dataset, 
        features=datasets.features.features.Features(
            {
                'ner_tags': datasets.Sequence(feature=datasets.ClassLabel(names=label_list)),
                'tokens': datasets.Sequence(feature=datasets.Value(dtype='string'))
            }))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_all_tokens = True

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

    train_dataset = tokenized_datasets['train']
    path_model = os.path.join(path, 'model', model_framework, 
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if model_framework == ModelFramework.PyTorch:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(label_list)) 
    if model_framework == ModelFramework.Tensorflow:
        model = TFAutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(label_list), from_pt=True)
    training_args = TrainingArguments(os.path.join(path_model, 'training_args'))
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    trainer.train()
    kili_print(f'Saving model to {path_model}')
    trainer.save_model(ensure_dir(path_model))


def huggingface_train_text_classification_single(
        api_key:str, assets:List[Dict], job: Dict, job_name:str,
        model_framework:str, model_name:str, path:str):
    '''
    Source: https://huggingface.co/docs/transformers/training
    '''
    path_dataset = os.path.join(path, 'dataset', 'data.json')
    kili_print(f'Downloading data to {path_dataset}')
    if os.path.exists(path_dataset):
        os.remove(path_dataset)
    job_categories = list(job['content']['categories'].keys())
    with open(ensure_dir(path_dataset), 'w') as handler:
        for asset in assets:
            response = requests.get(asset['content'], headers={
                'Authorization': f'X-API-Key: {api_key}',
            })
            label_category = asset['labels'][0]['jsonResponse'][job_name]['categories'][0]['name']
            handler.write(json.dumps({
                'text': response.text,
                'label': job_categories.index(label_category),
                }) + '\n')
    raw_datasets = datasets.load_dataset('json', 
        data_files=path_dataset,
        features=datasets.features.features.Features(
            {
                'label': datasets.ClassLabel(names=job_categories),
                'text': datasets.Value(dtype='string')
            }))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets['train']
    path_model = os.path.join(path, 'model', model_framework, 
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if model_framework == ModelFramework.PyTorch:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(job_categories)) 
    if model_framework == ModelFramework.Tensorflow:
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(job_categories), from_pt=True)
    training_args = TrainingArguments(os.path.join(path_model, 'training_args'))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    kili_print(f'Saving model to {path_model}')
    trainer.save_model(ensure_dir(path_model))



