import os

from termcolor import colored
from tqdm import tqdm

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


def get_assets(kili, project_id, label_types):
    total = kili.count_assets(project_id=project_id)
    first = 100
    assets = []
    for skip in tqdm(range(0, total, first)):
        assets += kili.assets(
            project_id=project_id,
            first=first,
            skip=skip,
            disable_tqdm=True,
            fields=[
                'id',
                'externalId',
                'content',
                'labels.createdAt',
                'labels.jsonResponse',
                'labels.labelType'])
    assets = [{
        **a,
        'labels': [
            l for l in sorted(a['labels'], key=lambda l: l['createdAt']) \
                if l['labelType'] in label_types
                ][-1:],
                } for a in assets]
    assets = [a for a in assets if len(a['labels']) > 0]
    return assets


def get_project(kili, project_id):
    projects = kili.projects(project_id=project_id, fields=['inputType', 'jsonInterface'])
    if len(projects) == 0:
        raise ValueError('no such project')
    input_type = projects[0]['inputType']
    jobs = projects[0]['jsonInterface'].get('jobs', {})
    return input_type, jobs


def kili_print(*args, **kwargs):
    print(colored('kili:', 'yellow', attrs=['bold']), *args, **kwargs)


def set_default(x, x_default, x_name, x_range):
    if x not in x_range:
            kili_print(f'defaulting to {x_name}={x_default}')
            x = x_default
    return x