from typing import Any, Dict

from typing_extensions import Literal

LabelingStatusT = Literal["LABELED", "UNLABELED"]
StatusIntT = Literal["TODO", "ONGOING", "LABELED", "REVIEWED", "DEFAULT"]
LabelTypeT = Literal["DEFAULT", "REVIEW"]
CommandT = Literal["train", "predict", "label_errors", "prioritize"]


AssetT = Dict[str, Any]
JobT = Dict[str, Any]
JobsT = Dict[str, JobT]
TrainingArgsT = Dict[str, Any]
