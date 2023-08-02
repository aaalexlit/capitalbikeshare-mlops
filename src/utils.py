from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    return get_project_root() / "data"