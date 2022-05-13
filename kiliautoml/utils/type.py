from typing import Any, Dict

from typing_extensions import Literal

AssetStatusT = Literal["TODO", "ONGOING", "LABELED", "TO_REVIEW", "REVIEWED"]
LabelTypeT = Literal["PREDICTION", "DEFAULT", "AUTOSAVE", "REVIEW", "INFERENCE"]
CommandT = Literal["train", "predict", "label_errors", "prioritize"]


AssetT = Dict[str, Any]
JobT = Dict[str, Any]
JobsT = Dict[str, JobT]
TrainingArgsT = Dict[str, Any]
