# src/model.py
"""
Model loading utilities for the Diffusion Engine.

Loads a single Stable Diffusion pipeline (base model) and configures
its scheduler for inference.
"""

from __future__ import annotations

import warnings

# Suppress conflicting diffusers dtype warnings (library in API transition)
warnings.filterwarnings("ignore", message=r".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=r".*dtype.*are not expected.*")

import torch
from diffusers import DiffusionPipeline, DDPMScheduler

from src.config import DEVICE, MODEL_ID


def _dtype_for(device: torch.device) -> torch.dtype:
    """Select optimal dtype for the target device."""
    return torch.float16 if device.type == "cuda" else torch.float32


def load_pipeline(device: torch.device = DEVICE) -> DiffusionPipeline:
    """
    Load and configure the diffusion pipeline.

    Uses DDPMScheduler to match the manual DDPM math in diffusion.py.
    """
    dtype = _dtype_for(device)

    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe