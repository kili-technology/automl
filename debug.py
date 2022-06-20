from main import kiliautoml

if __name__ == "__main__":
    # The command you want to debug
    cmd = "predict --project-id XXX --max-assets 40"
    kiliautoml(cmd.split(" "))
