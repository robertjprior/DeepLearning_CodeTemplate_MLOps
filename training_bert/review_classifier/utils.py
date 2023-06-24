import json
import numpy as np
import random

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

def save_dict(d, filepath):
    """Save dict to a json file."""
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)

def load_dict(filepath):
    """Load a dict from a json file."""
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d

# import tomli
# def load_toml(filepath):
#     with open(filepath, mode="rb") as fp:
#         toml_dict = tomli.load(fp)
#     return toml_dict