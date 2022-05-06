[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-brightgreen.svg)](https://flake8.pycqa.org/)


# Kili AutoML

AutoML is a lightweight library to create ML models in a data-centric AI way:

1. Label on [Kili](https://www.kili-technology.com)
2. **Train** a model with AutoML and evaluate its performance in one line of code
3. **Push** predictions to [Kili](https://www.kili-technology.com) to accelerate the labeling in one line of code
4. **Prioritize** labeling on [Kili](https://www.kili-technology.com) to label the data that will improve your model the most first

Iterate.

Once you are satisfied with the performance, in one line of code, **serve** the model and monitor the performance keeping a human in the loop with [Kili](https://www.kili-technology.com).

## Installation

```bash
git clone https://github.com/kili-technology/automl.git
cd automl
git submodule update --init
```

then
```bash
pip install -r requirements.txt -r utils/ultralytics/yolov5/requirements.txt
```

## Usage

We made AutoML very simple to use. The main methods are:

### Train a model

```bash
python train.py \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```

Retrieve the annotated data from the project and specialize the best model among the following ones on each task:

- Hugging Face (NER, Text Classification)
- YOLOv5 (Object Detection)
- spaCy (coming soon)
- Simple Transformers (coming soon)
- Catalyst (coming soon)
- XGBoost & LightGBM (coming soon)

Compute model loss to infer when you can stop labeling.

![Train a model](./images/train.png)

### Push predictions to Kili

```bash
python predict.py \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```

Use trained models to push pre-annotations onto unlabeled assets. Typically speeds up labeling by 10% with each iteration.

![Predict a model](./images/predict.png)

You can also use a model coming from another project, if they have the same ontology:
```bash
python predict.py \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID \
    --from-project $ANOTHER_KILI_PROJECT_ID
```

### Prioritize labeling on Kili

Where is the model confident or confused today?

```bash
python prioritize.py \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
    --sampling uncertainty
    --method least-confidence-sampling
```

How can we sample the optimal unlabeled data points for human review?

```bash
python prioritize.py \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
    --sampling diversity
    --method model-based-outlier
```

### Label errors on Kili
Note: for image classfication projects only.

```bash
python label_errors.py \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```


### Serve a model (coming soon)

```bash
python serve.py \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```

Serve trained models while pushing assets and predictions to [Kili](https://www.kili-technology.com) for continuous labeling. Allows monitoring the model drift.

![Serve a model](./images/serve.png)

## ML Tasks

AutoML currently supports the following tasks:

- NLP
  - [Named Entity Recognition](examples/ner.md)

## Disclaimer

AutoML is a utility library that trains and serves models. It is your responsibility to determine whether the model performance is high enough or not.

Don't hesitate to contribute!
