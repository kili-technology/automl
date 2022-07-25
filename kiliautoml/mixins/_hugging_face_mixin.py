# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
import os
from abc import ABCMeta
from datetime import datetime
from typing import Any, Dict, List, Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    TFAutoModelForTokenClassification,
    TrainingArguments,
)

from kiliautoml.utils.helpers import get_last_trained_model_path, kili_print
from kiliautoml.utils.path import PathHF
from kiliautoml.utils.type import (
    CategoryIdT,
    JobNameT,
    MLBackendT,
    MLTaskT,
    ModelNameT,
    ModelRepositoryT,
    ProjectIdT,
)


class HuggingFaceMixin(metaclass=ABCMeta):
    """
    Methods common to HuggingFace jobs
    """

    model_repository: ModelRepositoryT = "huggingface"

    @staticmethod
    def _get_tokenizer_and_model(ml_backend: MLBackendT, model_path: str, ml_task: MLTaskT):
        if ml_backend == "pytorch":
            tokenizer = AutoTokenizer.from_pretrained(model_path, from_pt=True)
            if ml_task == "NAMED_ENTITIES_RECOGNITION":
                model = AutoModelForTokenClassification.from_pretrained(model_path)
            elif ml_task == "CLASSIFICATION":
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                raise ValueError("unknown model task")
        elif ml_backend == "tensorflow":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if ml_task == "NAMED_ENTITIES_RECOGNITION":
                model = TFAutoModelForTokenClassification.from_pretrained(model_path)
            elif ml_task == "CLASSIFICATION":
                model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                raise ValueError("unknown model task")
        else:
            raise NotImplementedError
        return tokenizer, model

    def _get_tokenizer_and_model_from_name(
        self,
        model_name: ModelNameT,
        ml_backend: MLBackendT,
        label_list: List[CategoryIdT],
        ml_task: MLTaskT,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"num_labels": len(label_list), "id2label": dict(list(enumerate(label_list)))}
        if ml_backend == "pytorch":
            pass
        elif ml_backend == "tensorflow":
            kwargs.update({"from_pt": True})
        else:
            raise NotImplementedError

        if ml_task == "NAMED_ENTITIES_RECOGNITION":
            model = AutoModelForTokenClassification.from_pretrained(model_name, **kwargs)
        elif ml_task == "CLASSIFICATION":
            model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        else:
            raise NotImplementedError

        return tokenizer, model

    @classmethod
    def _extract_model_info(
        cls,
        job_name: JobNameT,
        project_id: ProjectIdT,
        model_path,
        from_project: Optional[ProjectIdT],
    ):
        if from_project is not None:
            if model_path is None:
                project_id = from_project
            else:
                kili_print(
                    "You have specified both a model path and a project id. "
                    "The model path will be used."
                )

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

        if split_path[-2] in ["pytorch", "tensorflow"]:
            ml_backend: MLBackendT = split_path[-2]  # type: ignore
            kili_print(f"ml-backend: {ml_backend}")
        else:
            raise ValueError("Unknown ml-backend")
        return model_path_res, cls.model_repository, ml_backend

    @staticmethod
    def _get_training_args(
        path_model,
        model_name,
        epochs: int,
        disable_wandb: bool,
        batch_size: int,
        additional_train_args_hg: Dict[str, Any],
    ):
        date = datetime.now().strftime("%Y-%m-%d_%H:%M")
        default_args = {
            "report_to": "wandb" if not disable_wandb else "none",
            "run_name": model_name + "_" + date,
        }
        default_args.update(additional_train_args_hg)
        training_args = TrainingArguments(
            PathHF.append_training_args_dir(path_model),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            **default_args,
        )
        return training_args
