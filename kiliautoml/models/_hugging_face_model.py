from kiliautoml.models._base_model import BaseInitArgs, KiliBaseModel


class HuggingFaceModel(KiliBaseModel):
    def __init__(self, base_init_args: BaseInitArgs) -> None:
        super().__init__(base_init_args)

    def fill_model_name(self, base_init_args: BaseInitArgs) -> None:
        pass
