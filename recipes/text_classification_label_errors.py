import os
import re

from cleanlab.filter import find_label_issues
from datasets import load_dataset
from kili.client import Kili
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tqdm import tqdm

MAX_ASSETS = 500

kili = Kili(api_key=os.getenv("KILI_API_KEY"))

dataset = load_dataset("ag_news")
ds_test = dataset["test"]
names = ds_test.features["label"].names


def to_key(name):
    return re.sub("[^0-9a-zA-Z]+", "_", name.lower())


def create_project_if_not_exists():
    projects = kili.projects(search_query="ag_news", fields=["id"])
    if len(projects) == 1:
        return projects[0]
    if len(projects) > 1:
        raise Exception("Duplicates")
    categories = {to_key(name): {"name": name} for name in names}
    json_interface = {
        "jobs": {
            "CLASSIFICATION_JOB": {
                "mlTask": "CLASSIFICATION",
                "content": {
                    "categories": categories,
                    "input": "radio",
                },
                "instruction": "Classify AG's news topic",
            }
        }
    }
    project = kili.create_project(
        title="ag_news",
        description=(
            "AG is a collection of more than 1 million news articles. News articles have been"
            " gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of"
            " activity. ComeToMyHead is an academic news search engine which has been running since"
            " July, 2004."
        ),
        input_type="TEXT",
        json_interface=json_interface,
    )
    project_id = project["id"]
    for i, text in enumerate(tqdm(ds_test["text"])):
        if i == MAX_ASSETS:
            break
        external_id = f"test[{i}]"
        kili.append_many_to_dataset(
            project_id=project_id, content_array=[text], external_id_array=[external_id]
        )
        kili.append_to_labels(
            label_asset_external_id=external_id,
            json_response={
                "CLASSIFICATION_JOB": {
                    "categories": [{"name": to_key(names[ds_test["label"][i]]), "confidence": 100}]
                }
            },
            project_id=project_id,
        )


if __name__ == "__main__":
    project = create_project_if_not_exists()
    classifier = Pipeline([("vect", CountVectorizer()), ("clf", MultinomialNB())])
    classifier.fit(X=ds_test["text"], y=ds_test["label"])
    pred_probs = classifier.predict_proba(ds_test["text"])
    ranked_label_issues = find_label_issues(
        labels=ds_test["label"], pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
    )
    for i in ranked_label_issues:
        if i >= MAX_ASSETS:
            continue
        prediction = np.argmax(pred_probs[i])
        external_id = f"test[{i}]"
        project_id = project["id"]
        labels = kili.labels(
            asset_external_id_in=[external_id], project_id=project_id, fields=["createdAt", "id"]
        )
        if len(labels) == 0:
            continue
        labels = sorted(labels, key=lambda d: d["createdAt"])
        issue = kili.append_to_issues(
            issue_number=0,
            label_id=labels[-1]["id"],
            object_mid="",
            type="QUESTION",
            external_id=external_id,
            project_id=project_id,
        )
        issue_id = issue["id"]
        current_label = names[ds_test["label"][i]]
        suggested_label = names[prediction]
        kili.append_to_comments(
            text=f"Suggestion: replace {current_label} with {suggested_label}",
            in_review=False,
            issue_id=issue_id,
        )
