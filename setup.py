from setuptools import setup

setup(
    name="kiliautoml",
    version="0.1.0",
    py_modules=["main"],
    install_requires=[
        "cleanlab>=2.0.0",
        "click",
        "datasets<=2.2",
        "evaluate",
        "seqeval",
        "kili",
        "jinja2",
        "nltk>=3.3",
        "numpy",
        "requests",
        "scikit-learn",
        "tabulate",
        "termcolor",
        "tqdm",
        "transformers",
        "img2vec-pytorch==1.0.1",
        "more-itertools",
        "typing_extensions",
        "wandb",
        "ipython",
        "ratelimit",
        "opencv-python",
        "detectron2 @ git+https://github.com/facebookresearch/detectron2.git",
        "shapely",
        "pydantic",
        "pytest-mock",
        "backoff",
        "loguru",
        # ################################ yolo
        "matplotlib>=3.2.2",
        "numpy>=1.18.5",
        "opencv-python>=4.1.2",
        "Pillow>=7.1.2",
        "PyYAML>=5.3.1",
        "requests>=2.23.0",
        "scipy>=1.4.1",
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "tqdm>=4.41.0",
        # Logging -------------------------------------
        "tensorboard>=2.4.1",
        # wandb
        # Plotting ------------------------------------
        "seaborn>=0.11.0",
        "thop",
    ],
    entry_points={
        "console_scripts": [
            "kiliautoml = main:kiliautoml",
        ],
    },
)
