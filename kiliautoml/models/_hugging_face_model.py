import requests
from bs4 import BeautifulSoup
from typing import Optional

from kiliautoml.utils.logging import logger
from kiliautoml.models._base_model import BaseInitArgs, KiliBaseModel, ModelConditions
from kiliautoml.utils.type import ModelNameT


class HuggingFaceModelConditions(ModelConditions):

    def _check_compatible_model(self, model_name: Optional[ModelNameT]) -> None:
        if model_name and model_name not in self.advised_model_names:
            # check if the model is a fill-mask model
            html = requests.get(f"https://huggingface.co/{model_name}").text
            soup = BeautifulSoup(html, "html.parser")
            if not (len([x for x in soup.find_all("span") if "Fill-Mask" in x]) > 1):
                raise ValueError(
                    f"Wrong model requested {model_name}. Try one of these models: \n "
                    f"{str(self.advised_model_names)} or any HuggingFace Fill-Mask model."
                )
            else:
                logger.warning(
                    f"{model_name} is not one of the advised models {self.advised_model_names}"
                )


class HuggingFaceModel(KiliBaseModel):
    def __init__(self, base_init_args: BaseInitArgs) -> None:
        super().__init__(base_init_args)

    def fill_model_name(self, base_init_args: BaseInitArgs) -> None:
        pass
