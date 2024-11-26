import pathlib


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent


def get_max_n_nodes(depth):
    return 2 ** (depth + 1) - 1
