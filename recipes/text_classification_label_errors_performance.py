import matplotlib.pyplot as plt
import os
import requests

from bert_sklearn import BertClassifier
from cleanlab.filter import find_label_issues
from datasets import load_dataset
from kili.client import Kili
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def performance_with_all_suggestions(name, config=None):
    dataset = load_dataset(name, config)
    ds_train = dataset["train"]
    ds_test = dataset["test"]
    names = ds_train.features["label"].names
    clf = Pipeline([("vect", CountVectorizer()), ("clf", MultinomialNB())])

    # Default
    clf.fit(X=ds_train["text"], y=ds_train["label"])
    y_true = ds_test["label"]
    y_pred = clf.predict(ds_test["text"])
    default_accuracy = accuracy_score(y_true, y_pred)
    print(f"Default accuracy: {default_accuracy}")

    # Fixed train
    y_train = ds_train["label"]
    pred_probs = clf.predict_proba(ds_train["text"])
    ranked_label_issues = find_label_issues(
        labels=ds_train["label"], pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
    )
    for i in ranked_label_issues[:100]:
        y_train[i] = np.argmax(pred_probs[i])
    clf.fit(X=ds_train["text"], y=y_train)
    y_pred = clf.predict(ds_test["text"])
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Fixed train accuracy: {accuracy}")

    # Fix train and test
    pred_probs = clf.predict_proba(ds_test["text"])
    ranked_label_issues = find_label_issues(
        labels=ds_test["label"], pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
    )
    for i in ranked_label_issues[:100]:
        y_true[i] = np.argmax(pred_probs[i])
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Fixed train & test accuracy: {accuracy}")


def performance_with_labelerrors_ground_truth():
    dataset = load_dataset("imdb")
    X_train = dataset["test"]["text"]
    y_train = dataset["test"]["label"]
    X_test = dataset["train"]["text"]
    y_test = dataset["train"]["label"]
    names = dataset["train"].features["label"].names

    clf = Pipeline([("vect", CountVectorizer()), ("clf", MultinomialNB())])
    clf.fit(X=X_train, y=y_train)
    default_accuracy = accuracy_score(y_test, clf.predict(X_test))
    print("base accuracy", default_accuracy)

    number_of_changes = 0
    for page in range(1, 100):
        response = requests.get(
            f"https://labelerrors.com/api/data?dataset=IMDB&page={page}&limit=10"
        )
        if response.status_code == 404:
            break
        data = response.json()
        for d in data:
            path = d["path"]
            response = requests.get(f"https://labelerrors.com/{path}")
            text = response.content.decode("utf-8")
            label_name = d["guessed_label"].lower()[:3]
            label = np.argwhere(np.array(names) == label_name)[0][0]
            index = np.argwhere(np.array(X_train) == text)
            if len(index) == 0:
                continue
            index = index[0][0]
            y_train[index] = label
            number_of_changes += 1
    clf.fit(X=X_train, y=y_train)
    fixed_accuracy = accuracy_score(y_test, clf.predict(X_test))
    print("fixed accuracy", fixed_accuracy)
    print("number of labels fixed in the train set", 100 * (number_of_changes / len(X_train)), "%")
    print("relative gain of accuracy", 100 * (fixed_accuracy / default_accuracy - 1), "%")


def performance_increase_simulation():
    dataset = load_dataset("imdb")
    X_train = dataset["test"]["text"]
    y_train = np.array(dataset["test"]["label"])
    X_test = dataset["train"]["text"]
    y_test = dataset["train"]["label"]

    accuracies = []
    clf = Pipeline([("vect", CountVectorizer()), ("clf", MultinomialNB())])
    clf.fit(X=X_train, y=y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    proportion_pct = 0.0
    print(proportion_pct, accuracy)
    proportions_pct = [proportion_pct]
    accuracies.append(accuracy)
    for i in range(0, int(0.2 * len(X_train)), int(0.01 * len(X_train))):
        index = slice(i, (i + int(0.01 * len(X_train)) - 1))
        y_train[index] = 1 - y_train[index]
        clf.fit(X=X_train, y=y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        proportion_pct = i / len(X_train) * 100
        print(proportion_pct, accuracy)
        proportions_pct.append(proportion_pct)
        accuracies.append(accuracy)
    plt.plot(proportions_pct, accuracies)
    plt.xlabel("number of wrong labels in the train set (%)")
    plt.ylabel("accuracy")
    plt.show()


def get_consensus_values():
    kili = Kili(api_key=os.getenv("KILI_ADMIN_API_KEY"))
    projects = kili.projects(
        as_generator=True,
        fields=["id", "consensusMark", "numberOfAssets", "numberOfRemainingAssets"],
        first=kili.count_projects(),
    )
    consensus = []
    for project in projects:
        if (
            project["consensusMark"] is None
            or project["numberOfAssets"] - project["numberOfRemainingAssets"] < 200
        ):
            continue
        consensus.append(100 * project["consensusMark"])
    consensus = np.array(consensus)
    print(np.mean(consensus))
    plt.hist(consensus, bins=20)
    plt.xlabel("project consensus (%)")
    plt.show()


def performance_increase_with_bert(name="imdb", config=None):
    dataset = load_dataset(name, config)
    ds_train = dataset["train"]
    ds_test = dataset["test"]
    y_true = ds_test["label"]

    # Default
    clf = Pipeline([("vect", CountVectorizer()), ("clf", MultinomialNB())])
    clf.fit(X=ds_train["text"], y=ds_train["label"])
    y_pred = clf.predict(ds_test["text"])
    default_accuracy = accuracy_score(y_true, y_pred)
    print(f"Default accuracy: {default_accuracy}")

    # Bert
    clf = BertClassifier()
    clf.fit(X=ds_train["text"], y=ds_train["label"])
    y_pred = clf.predict(ds_test["text"])
    bert_accuracy = accuracy_score(y_true, y_pred)
    print(f"Bert accuracy: {bert_accuracy}")


if __name__ == "__main__":
    # performance_with_all_suggestions("ag_news")
    # performance_with_all_suggestions("imdb")
    # performance_with_all_suggestions("tweet_eval", config="emoji")
    # performance_with_labelerrors_ground_truth()
    # performance_increase_simulation()
    # get_consensus_values()
    performance_increase_with_bert()
