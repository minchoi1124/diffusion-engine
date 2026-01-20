# src/config.py
"""
Central configuration for the Diffusion Engine.

This file defines:
- Reproducibility settings
- Device selection
- Diffusion hyperparameters
- Default image settings
- Common paths
"""

import os
import random
import numpy as np
import torch

# Reproducibility

GLOBAL_SEED = 1124

def seed_everything(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Device

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()

# Diffusion parameters

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# Image defaults

IMAGE_SIZE = 512

# Model identifiers

MODEL_ID = "CompVis/stable-diffusion-v1-4"

# Paths

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def ensure_dirs():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)