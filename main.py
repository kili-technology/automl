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
def group():
    pass


group.add_command(train, name="train")
group.add_command(predict, name="predict")
group.add_command(label_errors, name="label_errors")
group.add_command(prioritize, name="prioritize")


if __name__ == "__main__":
    group()
