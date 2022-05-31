import os
import pickle

from datasets import load_dataset
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import umap
import umap.plot


MAX_ASSETS = 1000

dataset = load_dataset("ag_news")
ds_train = dataset["train"]
ds_test = dataset["test"]
names = ds_train.features["label"].names


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_hidden_states=True,
)


def text_to_embedding(text):
    text = ds_train["text"][0]
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    return torch.mean(token_vecs, dim=0).cpu().detach().numpy()


path = "./tmp.pkl"
if os.path.exists(path):
    with open(path, "rb") as handler:
        embeddings = pickle.load(handler)
else:
    embeddings = [text_to_embedding(text) for text in tqdm(ds_test["text"][:MAX_ASSETS])]
    with open(path, "wb") as handler:
        pickle.dump(embeddings, handler)

mapper = umap.UMAP().fit(embeddings)

umap.plot.points(mapper, labels=np.array(ds_test["label"][:MAX_ASSETS]))
# plt.savefig("FigureName.png")

# import umap
# import umap.plot
# from sklearn.datasets import load_digits

# digits = load_digits()

# mapper = umap.UMAP().fit(digits.data)
# umap.plot.points(mapper, labels=digits.target)

plt.show(block=True)
