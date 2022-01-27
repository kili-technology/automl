import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import click
from kili.client import Kili

from utils.constants import ContentInput, HOME, InputType, MLTask, \
    ModelFramework, ModelName, ModelRepository
from utils.helpers import get_assets, get_project, set_default
from utils.huggingface.train import huggingface_train_ner, huggingface_train_text_classification_single


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def train_ner(
        api_key, assets, job, job_name, 
        model_framework, model_name, model_repository, project_id):
    model_repository = set_default(model_repository, ModelRepository.HuggingFace, 
        'model_repository', [ModelRepository.HuggingFace])
    path = os.path.join(HOME, project_id, job_name, model_repository)
    if model_repository == ModelRepository.HuggingFace:
        model_framework = set_default(model_framework, ModelFramework.PyTorch, 
            'model_framework', [ModelFramework.PyTorch, ModelFramework.Tensorflow])
        model_name = set_default(model_name, ModelName.BertBaseMultilingualCased, 
            'model_name', [ModelName.BertBaseMultilingualCased])
        huggingface_train_ner(
            api_key, assets, job, job_name, model_framework, model_name, path)


def train_text_classification_single(
        api_key, assets, job, job_name, 
        model_framework, model_name, model_repository, project_id):
    model_repository = set_default(model_repository, ModelRepository.HuggingFace, 
        'model_repository', [ModelRepository.HuggingFace])
    path = os.path.join(HOME, project_id, job_name, model_repository)
    if model_repository == ModelRepository.HuggingFace:
        model_framework = set_default(model_framework, ModelFramework.PyTorch, 
            'model_framework', [ModelFramework.PyTorch, ModelFramework.Tensorflow])
        model_name = set_default(model_name, ModelName.BertBaseMultilingualCased, 
            'model_name', [ModelName.BertBaseMultilingualCased])
        huggingface_train_text_classification_single(
            api_key, assets, job, job_name, model_framework, model_name, path)




@click.command()
@click.option('--api-key', default=None, help='Kili API Key')
@click.option('--model-framework', default=None, help='Model framework (eg. pytorch, tensorflow)')
@click.option('--model-name', default=None, help='Model name (eg. bert-base-cased)')
@click.option('--model-repository', default=None, help='Model repository (eg. huggingface)')
@click.option('--project-id', default=None, help='Kili project ID')
def main(api_key: str, model_framework: str, model_name: str, model_repository: str, project_id: str):
    kili = Kili(api_key=api_key)
    input_type, jobs = get_project(kili, project_id)
    assets = get_assets(kili, project_id)
    for job_name, job in jobs.items():
        content_input = job.get('content', {}).get('input')
        ml_task = job.get('mlTask')
        tools = job.get('tools')
        if content_input == ContentInput.Radio \
                and input_type == InputType.Text \
                and ml_task == MLTask.Classification:
            train_text_classification_single(
                api_key, assets, job, job_name, 
                model_framework, model_name, model_repository, project_id)
        if content_input == ContentInput.Radio \
                and input_type == InputType.Text \
                and ml_task == MLTask.NamedEntitiesRecognition:
            train_ner(
                api_key, assets, job, job_name, 
                model_framework, model_name, model_repository, project_id)
            
    



    


if __name__ == '__main__':
    main()