from typing_extensions import Literal

labeling_statusT = Literal["LABELED", "UNLABELED"]
status_inT = Literal["TODO", "ONGOING", "LABELED", "REVIEWED", "DEFAULT"]
label_typeT = Literal["DEFAULT", "REVIEW"]
