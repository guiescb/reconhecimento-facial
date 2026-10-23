"""Configuracao, semente aleatoria e registro do ambiente (reprodutibilidade)."""
import platform
import random
import sys

import numpy as np
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def environment_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }
    try:
        import sklearn
        info["scikit_learn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info
