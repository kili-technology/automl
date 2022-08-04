"""It's not possible to use the python debugger with the command kiliautoml.

This file enables to use the the debugger as follows:
python$
"""
from click.testing import CliRunner

import main

if __name__ == "__main__":
    # The command you want to debug
    cmd = "train --project-id XXX"
    cmd = cmd.replace("  ", " ")

    runner = CliRunner()
    result = runner.invoke(
        main.kiliautoml,
        cmd.split(" "),
    )
