from setuptools import setup

setup(
    name="kiliautoml",
    version="0.1.0",
    py_modules=["main"],
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
        "ipython",
        "ratelimit",
    ],
    entry_points={
        "console_scripts": [
            "kiliautoml = main:kiliautoml",
        ],
    },
)
