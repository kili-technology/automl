print("Loading KiliAutoML...")

import sys

import click
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=False)
from commands.label_errors import main as label_errors
from commands.predict import main as predict
from commands.prioritize import main as prioritize
from commands.train import main as train


@click.group()
def kiliautoml():
    pass


kiliautoml.add_command(train, name="train")
kiliautoml.add_command(predict, name="predict")
kiliautoml.add_command(label_errors, name="label_errors")
kiliautoml.add_command(prioritize, name="prioritize")


if __name__ == "__main__":
    kiliautoml()
