import json
import os
from typing import Any, Dict, List
from typing_extensions import TypedDict
from warnings import warn

from nltk import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
import requests
from tqdm.auto import tqdm

from utils.helpers import ensure_dir, kili_print


def kili_assets_to_hf_ner_dataset(
    api_key: str,
    job: Dict,
    job_name: str,
    path_dataset: str,
    assets: List[Dict],
    clear_dataset_cache: bool,
):

    if clear_dataset_cache and os.path.exists(path_dataset):
        kili_print("Dataset cache for this project is being cleared.")
        os.remove(path_dataset)

    job_categories = list(job["content"]["categories"].keys())
    label_list = ["O"] + ["B-" + jc for jc in job_categories] + ["I-" + jc for jc in job_categories]

    labels_to_ids = {label: i for i, label in enumerate(label_list)}

    if os.path.exists(path_dataset) and clear_dataset_cache:
        os.remove(path_dataset)
    if not os.path.exists(path_dataset):
        with open(ensure_dir(path_dataset), "w") as handler:
            for asset in tqdm(assets):
                write_asset(api_key, job_name, labels_to_ids, handler, asset)

    return label_list


def write_asset(api_key, job_name, labels_to_ids, handler, asset):
    response = requests.get(
        asset["content"],
        headers={
            "Authorization": f"X-API-Key: {api_key}",
        },
    )
    text = response.text
    if (
        job_name not in asset["labels"][0]["jsonResponse"]
    ):  # always taking the first label (for now)
        asset_id = asset["id"]
        warn(f"${asset_id}: No annotation for job ${job_name}")
        return
    annotations = asset["labels"][0]["jsonResponse"][job_name]["annotations"]
    sentences = sent_tokenize(text)
    offset = 0
    for sentence_tokens in TreebankWordTokenizer().span_tokenize_sents(sentences):
        tokens = []
        ner_tags = []
        for start_without_offset, end_without_offset in sentence_tokens:
            start, end = (
                start_without_offset + offset,
                end_without_offset + offset,
            )
            token_annotations = [
                a
                for a in annotations
                if a["beginOffset"] <= start and a["beginOffset"] + len(a["content"]) >= end
            ]
            if len(token_annotations) > 0:
                category = token_annotations[0]["categories"][0]["name"]
                label = (
                    "B-" + category
                    if token_annotations[0]["beginOffset"] == start
                    else "I-" + category
                )
            else:
                label = "O"
            tokens.append(text[start:end])
            ner_tags.append(labels_to_ids[label])
        handler.write(
            json.dumps(
                {
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
            )
            + "\n"
        )
        offset = offset + sentence_tokens[-1][1] + 1


class KiliNerAnnotations(TypedDict):
    beginOffset: Any
    content: Any
    endOffset: Any
    categories: Any


def predicted_tokens_to_kili_annotations(
    text: str,
    predicted_label: List[str],
    predicted_proba: List[float],
    tokens: List[str],
    null_category: str,
    offset_in_text: int,
) -> List[KiliNerAnnotations]:
    """
    Format token predictions into a the kili format.
    :param: text:
    """

    kili_annotations: List[KiliNerAnnotations] = []
    offset_in_sentence = 0
    for label, proba, token in zip(predicted_label, predicted_proba, tokens):
        if token in [
            "[CLS]",
            "[SEP]",
            "[UNK]",
        ]:  # special BERT tokens that should ignored at inference time
            continue
        if token.startswith(
            "##"
        ):  # number tokens annotation should be ignored when aligning categories
            token = token.replace("##", "")

        text_remaining = text[offset_in_sentence:]
        ind = text_remaining.find(token)
        if ind == -1:
            raise Exception(f"token {token} not found in text {text_remaining}")

        offset_in_sentence += ind

        if label != null_category:
            is_i_tag = label.startswith("I-")
            c_kili = label.replace("B-", "").replace("I-", "")

            ann_ = {
                "beginOffset": offset_in_text + offset_in_sentence,
                "content": token,
                "endOffset": offset_in_text + offset_in_sentence + len(token),
                "categories": [{"name": c_kili, "confidence": int(proba * 100)}],
            }
            ann = KiliNerAnnotations(
                beginOffset=ann_["beginOffset"],
                content=ann_["content"],
                endOffset=ann_["endOffset"],
                categories=ann_["categories"],
            )

            if (
                len(kili_annotations)
                and ann["categories"][0]["name"] == kili_annotations[-1]["categories"][0]["name"]
                and (ann["beginOffset"] == kili_annotations[-1]["endOffset"] or is_i_tag)
            ):
                # merge with previous if same category and contiguous offset and onset:
                kili_annotations[-1]["endOffset"] = ann["endOffset"]
                kili_annotations[-1]["content"] += ann["content"]

            else:
                kili_annotations.append(ann)

        offset_in_sentence += len(token)

    return kili_annotations
