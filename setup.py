from setuptools import setup

setup(
    name="kiliautoml",
    version="0.1.0",
    py_modules=["train", "predict", "label_errors"],
    install_requires=[
        "Click",
        "cleanlab>=2.0.0",
        "click",
        "datasets",
        "kili",
        "jinja2",
        "nltk>=3.3",
        "numpy",
        "requests",
        "scikit-learn",
        "tabulate",
        "tensorflow",
        "termcolor",
        "tqdm",
        "transformers",
        "img2vec-pytorch==1.0.1",
        "more-itertools",
        "typing_extensions",
        "wandb==0.12.10",
    ],
    entry_points={
        "console_scripts": [
            "kiliautoml_train = train:main",
            "kiliautoml_predict = predict:main",
            "kiliautoml_label_errors = label_errors:main",
        ],
    },
)
