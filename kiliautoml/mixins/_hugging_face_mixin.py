# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
from abc import ABCMeta
import os
from typing import List

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    TFAutoModelForTokenClassification,
)

from kiliautoml.utils.constants import ModelFramework, ModelRepository, MLTask, ModelName
from kiliautoml.utils.helpers import get_last_trained_model_path, kili_print


class HuggingFaceMixin(metaclass=ABCMeta):
    """
    Methods common to HuggingFace jobs
    """

    model_repository = ModelRepository.HuggingFace

    @staticmethod
    def _get_tokenizer_and_model(model_framework: ModelFramework, model_path: str, ml_task: MLTask):
        if model_framework == ModelFramework.PyTorch:
            tokenizer = AutoTokenizer.from_pretrained(model_path, from_pt=True)
            if ml_task == MLTask.NamedEntityRecognition:
                model = AutoModelForTokenClassification.from_pretrained(model_path)
            elif ml_task == MLTask.Classification:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                raise ValueError("unknown model task")
        elif model_framework == ModelFramework.Tensorflow:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if ml_task == MLTask.NamedEntityRecognition:
                model = TFAutoModelForTokenClassification.from_pretrained(model_path)
            elif ml_task == MLTask.Classification:
                model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                raise ValueError("unknown model task")
        else:
            raise NotImplementedError
        return tokenizer, model

    def _get_tokenizer_and_model_from_name(
        self,
        model_name: ModelName,
        model_framework: ModelFramework,
        label_list: List[str],
        ml_task: MLTask,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"num_labels": len(label_list), "id2label": dict(list(enumerate(label_list)))}
        if model_framework == ModelFramework.PyTorch:
            pass
        elif model_framework == ModelFramework.Tensorflow:
            kwargs.update({"from_pt": True})
        else:
            raise NotImplementedError

        if ml_task == MLTask.NamedEntityRecognition:
            model = AutoModelForTokenClassification.from_pretrained(model_name, **kwargs)
        elif ml_task == MLTask.Classification:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        else:
            raise NotImplementedError

        return tokenizer, model

    @classmethod
    def _extract_model_info(cls, job_name, project_id, model_path):
        model_path_res = get_last_trained_model_path(
            job_name=job_name,
            project_id=project_id,
            model_path=model_path,
            project_path_wildcard=["*", "model", "*", "*"],
            weights_filename="pytorch_model.bin",
        )
        split_path = os.path.normpath(model_path_res).split(os.path.sep)
        if split_path[-4] != cls.model_repository:
            raise ValueError("Inconsistent model base repository")

        if split_path[-2] in [ModelFramework.PyTorch, ModelFramework.Tensorflow]:
            model_framework = split_path[-2]
            kili_print(f"Model framework: {model_framework}")
        else:
            raise ValueError("Unknown model framework")
        return model_path_res, cls.model_repository, model_framework
