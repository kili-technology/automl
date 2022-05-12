from typing_extensions import Literal

LabelingStatusT = Literal["LABELED", "UNLABELED"]
StatusIntT = Literal["TODO", "ONGOING", "LABELED", "REVIEWED", "DEFAULT"]
LabelTypeT = Literal["DEFAULT", "REVIEW"]
CommandT = Literal["train", "predict", "label_errors", "prioritize"]
