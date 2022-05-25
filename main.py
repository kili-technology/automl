import click

from commands.label_errors import main as label_errors
from commands.predict import main as predict
from commands.train import main as train


@click.group()
def group():
    pass


group.add_command(train, name="train")
group.add_command(predict, name="predict")
group.add_command(label_errors, name="label_errors")


if __name__ == "__main__":
    group()
