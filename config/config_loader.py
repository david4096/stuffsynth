import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
