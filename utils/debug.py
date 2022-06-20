"""It's not possible to use the python debugger with the command kiliautoml.

So this fife enables to use the command
python -m pdb utils.debug.py
"""
from main import kiliautoml

if __name__ == "__main__":
    # The command you want to debug
    cmd = "predict --project-id XXX --max-assets 40"
    kiliautoml(cmd.split(" "))
