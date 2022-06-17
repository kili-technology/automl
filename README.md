[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-brightgreen.svg)](https://flake8.pycqa.org/)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

# Kili AutoML

AutoML is a lightweight library to create ML models in a data-centric AI way:

1. Label on [Kili](https://www.kili-technology.com)
2. **Train** a model with AutoML and evaluate its performance in one line of code
3. **Push** predictions to [Kili](https://www.kili-technology.com) to accelerate the labeling in one line of code
4. **Prioritize** labeling on [Kili](https://www.kili-technology.com) to label the data that will improve your model the most first

Iterate.

Once you are satisfied with the performance, in one line of code, **serve** the model and monitor the performance keeping a human in the loop with [Kili](https://www.kili-technology.com).

## Quickstart

You can try automl on a simple image classification project with this notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kili-technology/automl/blob/main/notebooks/image_classification.ipynb)

## Installation
Creating a new conda or virtualenv before cloning is recommended because we install a lot of packages:

```bash
conda create --name automl python=3.7
conda activate automl
```

```bash
git clone https://github.com/kili-technology/automl.git
cd automl
git submodule update --init
```

then install the requirements:

```bash
pip install -r kiliautoml/utils/ultralytics/yolov5/requirements.txt
pip install -e .
```

## Usage

We made AutoML very simple to use. The following sections detail how to call the main methods.

### Train a model

We train the model with the following command line:

```bash
kiliautoml train \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```

By default, the library uses [Weights and Biases](https://wandb.ai/site) to track the training and the quality of the predictions.
The model is then stored in the cache of the AutoML library in `HOME/.cache/kili/automl`.
Kili automl training does the following:
* Selects the models related to the tasks declared in the project ontology.
* Retrieve Kili's asset data and convert it into the input format for each model.
* Finetunes the model on the input data.
* Outputs the model loss.

Here are the supported ML frameworks and the tasks they are used for.

- Hugging Face (NER, Text Classification)
- YOLOv5 (Object Detection)
- spaCy (coming soon)
- Simple Transformers (coming soon)
- Catalyst (coming soon)
- XGBoost & LightGBM (coming soon)

Compute model loss to infer when you can stop labeling.

![Train a model](./images/train.png)

### Push predictions to Kili

Once trained, the models are used to predict the labels, add preannotations on the assets that have not yet been labeled by the annotators. The annotators can then validate or correct the preannotations in the Kili user interface.

```bash
kiliautoml predict \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```

Using trained models to push pre-annotations onto unlabeled assets typically speeds up labeling by 10%.

![Predict a model](./images/predict.png)

You can also use a model coming from another project, if they have the same ontology:
```bash
kiliautoml predict \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID \
    --from-project $ANOTHER_KILI_PROJECT_ID
```

### Prioritize labeling on Kili

Once roughly 10 percent of the assets in a project have been labeled, it is possible to prioritize the remaining assets to be labeled on the project in order to prioritize the assets that will best improve the performance of the model.

```bash
kiliautoml prioritize \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```

This command will change the priority queue of the assets to be labeled.
To do this, AutoML uses a mix between diversity sampling and uncertainty sampling.

### Label errors on Kili
Note: for image classification projects only.

The error is human, fortunately there are methods to detect potential annotation problems. `label_errors.py` allows to identify potential problems and create a 'potential_label_error' filter on the project's asset exploration view:

```bash
kiliautoml label_errors \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```


## ML Tasks

AutoML currently supports the following tasks:

- Natural Language Processing (NLP)
  - [Named Entity Recognition](examples/ner.md)
  - Text Classification
- Image
  - Object detection
  - Image Classification

## Demos

You can test the features of AutoML with these notebooks:

- Natural Language Processing (NLP)
  - Named Entity Recognition
  - [Text Classification](https://colab.research.google.com/github/kili-technology/automl/blob/main/notebooks/text_classification.ipynb)
- Image
  - Object detection
  - [Image Classification](https://colab.research.google.com/github/kili-technology/automl/blob/main/notebooks/image_classification.ipynb)

## Disclaimer

AutoML is a utility library that trains and serves models. It is your responsibility to determine whether the model performance is high enough or not.

Don't hesitate to contribute!
