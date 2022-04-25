from termcolor import colored


def kili_print(*args, **kwargs) -> None:
    print(colored("kili:", "yellow", attrs=["bold"]), *args, **kwargs)
