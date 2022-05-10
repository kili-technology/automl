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

We train the model with just one line of code:

```bash
python train.py \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```

By default, the library uses wandb to track the training and to track the quality of the predictions.
The model is then stored in the cache of the AutoML library in HOME/.cache/kili/automl
We automatically choose a state of the art model corresponding to the type of ML task.
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

After training the model with the above python train.py command, we can then predict the labels and add preannotations on the assets that have not yet been labeled by the annotators. The annotators will then only have to validate or correct the preannotations.

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

Once roughly 10 percent of the assets in a project have been labeled, it is possible to prioritize the remaining assets to be labeled on the project in order to prioritize the assets that will best improve the performance of the model.

```bash
python prioritize.py \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```

This command will change the priority queue of the assets to be labeled.
To do this, AutoML uses a mix between diversity sampling and uncertainty sampling.

### Label errors on Kili
Note: for image classification projects only.

The error is human, fortunately there are methods to detect potential annotation problems. label_errors.py allows to identify potential problems and create a 'potential_label_error' filter on the project's asset exploration view:

```bash
python label_errors.py \
    --api-key $KILI_API_KEY \
    --project-id $KILI_PROJECT_ID
```


## Disclaimer

AutoML is a utility library that trains and serves models. It is your responsibility to determine whether the model performance is high enough or not.

Don't hesitate to contribute!
