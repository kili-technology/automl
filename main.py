import click

print("Loading KiliAutoML...")
from commands.label_errors import main as label_errors  # noqa: E402
from commands.predict import main as predict  # noqa: E402
from commands.prioritize import main as prioritize  # noqa: E402
from commands.train import main as train  # noqa: E402


@click.group()
def group():
    pass


group.add_command(train, name="train")
group.add_command(predict, name="predict")
group.add_command(label_errors, name="label_errors")
group.add_command(prioritize, name="prioritize")


if __name__ == "__main__":
    group()
