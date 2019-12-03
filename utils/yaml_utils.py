import yaml
from pathlib import Path


def read(path):
    with Path(path).open('r') as file:
        params = yaml.load(file, Loader=yaml.Loader)
    return params


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        yaml.dump(data, file)
