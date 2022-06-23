"""It's not possible to use the python debugger with the command kiliautoml.

This file enables to use the the debugger as follows:
python -m pdb utils/debug.py
"""
from main import kiliautoml

if __name__ == "__main__":
    # The command you want to debug
    cmd = "predict --project-id XXX --max-assets 40"
    kiliautoml(cmd.split(" "))
