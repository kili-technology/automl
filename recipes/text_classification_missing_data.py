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
ds_train = dataset["train"]
ds_test = dataset["test"]
names = ds_test.features["label"].names
