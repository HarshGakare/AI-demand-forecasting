import sys
from pathlib import Path

import numpy 
import pandas
import sklearn
import xgboost
import optuna
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def test_python_version():
    assert sys.version_info >= (3,11)
    
def test_project_folder():
    required = [
        "data",
        "notebooks",
        "src",
         "api",
        "dashboard",
        "sql",
        "models",
        "tests",
        "config",
    ]
    
    for folder in required:
        assert (PROJECT_ROOT / folder).exists()

def test_configuration():
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    assert config["project"]["random_seed"] == 42
    assert config["data"]["target_column"] == "Sales"
    assert config["training"]["tuning_trials"] > 0
    

def test_core_libraries():
    assert numpy.__version__
    assert pandas.__version__
    assert sklearn.__version__
    assert xgboost.__version__
    assert optuna.__version__
    
    
    