print("Loading KiliAutoML...")

import sys
from platform import python_version

version = python_version()
if ("3.7." not in version) and ("3.8." not in version):
    print("KiliAutoML requires Python 3.7.x or 3.8.x ")
    print("You are running Python {}".format(version))
    print("Please create a new virtual environment and install KiliAutoML")
    print("https://github.com/kili-technology/automl#installation")
    sys.exit(1)

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
