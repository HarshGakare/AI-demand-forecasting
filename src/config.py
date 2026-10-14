from pathlib import Path
import yaml

PROJECT_ROOT = path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        raise FileExistsError(
            f'Configuration file not found: {CONFIG_PATH}'
        )
        
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    if not isinstance(config, dict):
        raise ValueError(
            "COnfiguration must contain a Yaml mapping."
        )
        
    return config

CONFIG = load_config()