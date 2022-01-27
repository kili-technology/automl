# Kili AutoML

AutoML is a lightweight library providing three main features:

- training of models on data coming from a Kili project
- pushing prediction to a Kili project
- serving a model with human in the loop

## Install

```bash
git clone https://github.com/kili-technology/automl.git
cd automl
pip install -r requirements.txt
```

## Get started

```bash
python train.py \
    --api-key $KILI_API_KEY \
    --project-id ckysuic0y0ldc0lvoeltld164
python predict.py \
    --api-key $KILI_API_KEY \
    --project-id ckysuic0y0ldc0lvoeltld164
python serve.py \
    --api-key $KILI_API_KEY \
    --project-id ckysuic0y0ldc0lvoeltld164
```
