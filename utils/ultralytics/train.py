from datetime import datetime
import json
import os
from pyexpat import features
import shutil
from typing import Dict, List

import requests
import torch

from utils.constants import ModelFramework
from utils.helpers import ensure_dir, kili_print
from utils.ultralytics.yolov5.train import run



def ultralytics_train_yolov5(
        api_key:str, assets:List[Dict], job: Dict, job_name:str,
        model_framework:str, model_name:str, path:str) -> float:
    '''
    Source: https://huggingface.co/docs/transformers/training
    '''
    kili_print(job_name)
    path_dataset = os.path.join(path, 'dataset')
    if os.path.exists(path_dataset):
        shutil.rmtree(path_dataset)
    path_images = os.path.join(path_dataset, 'images', 'train')
    kili_print(f'Downloading data to {path_dataset}')
    if os.path.exists(path_images):
        os.remove(path_images)
    job_categories = list(job['content']['categories'].keys())
    for asset in assets:
        img_data = requests.get(asset['content'], headers={
                'Authorization': f'X-API-Key: {api_key}',
            }).content
        with open(ensure_dir(os.path.join(path_images, asset['id'] + '.jpg')), 'wb') as handler:
            handler.write(img_data)
    path_labels = os.path.join(path_dataset, 'labels', 'train')
    os.makedirs(path_labels, exist_ok=True)
    for asset in assets:
        with open(ensure_dir(os.path.join(path_labels, asset['id'] + '.txt')), 'w') as handler:
            json_response = asset['labels'][0]['jsonResponse']
            for job in json_response.values():
                for annotation in job.get('annotations', []):
                    name = annotation['categories'][0]['name']
                    category = job_categories.index(name)
                    bounding_poly = annotation.get('boundingPoly', [])
                    if len(bounding_poly) < 1:
                        continue
                    if 'normalizedVertices' not in bounding_poly[0]:
                        continue
                    normalized_vertices = bounding_poly[0]['normalizedVertices']
                    x_s = [vertice['x'] for vertice in normalized_vertices]
                    y_s = [vertice['y'] for vertice in normalized_vertices]
                    x_min, y_min = min(x_s), min(y_s)
                    x_max, y_max = max(x_s), max(y_s)
                    _x_, _y_ = (x_max + x_min) / 2, (y_max + y_min) / 2
                    _w_, _h_ = x_max - x_min, y_max - y_min
                    handler.write(f'{category} {_x_} {_y_} {_w_} {_h_}\n')
    # TODO: write kili.yaml
    run(data='kili.yaml')
    
