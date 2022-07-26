"""It's not possible to use the python debugger with the command kiliautoml.

This file enables to use the the debugger as follows:
python -m pdb utils/debug.py
"""
from click.testing import CliRunner

import main

if __name__ == "__main__":
    # The command you want to debug
    cmd = "label_errors --project-id cl5tqv0xkcv6b0nvr3qng5tqx  --asset-status-in LABELED"
    cmd = cmd.replace("  ", " ")

    runner = CliRunner()
    result = runner.invoke(
        main.kiliautoml,
        cmd.split(" "),
    )
