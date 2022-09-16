# Supported Models

Here are the supported ML backends and the tasks they are used for:

- Hugging Face (NER, Text Classification)
- YOLOv5 (Object Detection)
- Detectron2 (Semantic Segmentation)
- spaCy (coming soon)

For NLP tasks like NER or Text Classification you can use any Fill-Mask model on the HuggingFace Hub. But be aware that some additional install might be necessary for some models.

To choose your training model, you can use the following command:

```
kiliautoml train --api-key $KILI_API_KEY --project-id $PROJECT_ID --model-name your_model_name
```

## Advised models

We recommend to use some models that we have tested and are functional without any additional install. To see the recommended models for your project you can use this command:

```
kiliautoml advise --api-key $KILI_API_KEY --project-id $PROJECT_ID
```
