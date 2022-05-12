# Named Entity Recognition Recommendations

Named Entity Recognition (NER) is an NLP task that seeks to locate and classify relevant spans of text according to different entity types such as person names, organizations, locations...

**Example of NER:**

<ins>Sentence:</ins> François-Xavier Leduc is the CEO of Kili.

<ins>Result:</ins> [François-Xavier Leduc]\(*Person*) is the CEO of [Kili]\(*Company*).

## SOTA

The current state-of-the-art approaches are based on deep neural networks, mainly transformer models. These models are pretrained on lots of data using a semi-supervised approach and can be adapted to most of NLP tasks by adding a task-specific layer at the end of the model that is trained on specific task data.

Kili's AutoML pipeline follows this paradigm and uses BERT pre-trained models as the backbone of the NER model.

## Supported models

Kili's AutoML pipeline supports the following pre-trained models, available on [HuggingFace](https://huggingface.co/):

- [bert-base-multinlingual-cased](https://huggingface.co/bert-base-multilingual-cased): supports any language (performance might be low for low-ressource languages)
- [distilbert-base-cased](https://huggingface.co/distilbert-base-cased): only supports text in English

## Good practices for AutoML NER training

### Training set size

The size of the training dataset depends a lot on the task and on the data. There is no minimum dataset size from which we are sure that the model will work well. Nevertheless, for a good start, we recommend that you label at least 500 examples per entity type in your training set. Note that some samples can count for several entity types since several entity types can exist in one sample. In total, the training dataset should have at least 1000 samples.

Try to have samples as diverse as possible and keep a good balance between entity types. Prioritize the labeling of samples where entities and words are not present in the already-labeled data.

If you don't get the desired performance, do not hesitate to increase the size of the training dataset and label more data.

### Test of the model

Make sure to keep aside at least 10% of your annotated data to test the final model.
