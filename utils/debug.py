"""It's not possible to use the python debugger with the command kiliautoml.

This file enables to use the the debugger as follows:
python -m pdb utils/debug.py
"""
from click.testing import CliRunner

import main

if __name__ == "__main__":
    # The command you want to debug
    cmd = "predict  --project-id cl2k7tz4a02kg0lvrcgs71tzm --asset-status-in LABELED,REVIEWED"
    cmd = cmd.replace("  ", " ")

    runner = CliRunner()
    result = runner.invoke(
        main.kiliautoml,
        cmd.split(" "),
    )
