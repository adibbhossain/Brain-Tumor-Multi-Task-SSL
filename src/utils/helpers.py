import os
import random
import yaml
import numpy as np
import torch

def load_config(config_path: str = "configs/default_config.yaml") -> dict:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int = 42) -> None:
    """Ensures reproducibility by setting random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_device(device_setting: str = "auto") -> torch.device:
    """Resolves PyTorch torch.device based on availability and config string."""
    if device_setting == "auto" or device_setting is None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                return torch.device("cuda:1")
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(device_setting)

def ensure_dir(path: str) -> None:
    """Ensures parent directory exists."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
