# src/sampler.py
"""
Sampling engine for diffusion inference.

This module exposes a reusable sampling interface built on top of:
- a diffusion pipeline (UNet + VAE + tokenizer/text encoder + scheduler)
- DiffusionCore (math-only diffusion mechanics)

It implements classifier-free guidance (CFG) sampling as the primary
inference strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from PIL import Image

from src.config import DEVICE, IMAGE_SIZE, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, seed_everything
from src.diffusion import DiffusionCore


@dataclass(frozen=True)
class SampleConfig:
    """Configuration for a sampling run."""
    num_inference_steps: int = NUM_INFERENCE_STEPS
    guidance_scale: float = GUIDANCE_SCALE
    height: int = IMAGE_SIZE
    width: int = IMAGE_SIZE
    seed: int = 1124


def _autocast_if_cuda(device: torch.device):
    """
    Use autocast only when running on CUDA.
    Keeps logs clean on Mac/CPU while still enabling speedups on GPU.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    # no-op context manager
    from contextlib import nullcontext
    return nullcontext()


@torch.no_grad()
def encode_prompt(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text prompts into conditioning embeddings.

    Returns:
        (cond_embeds, uncond_embeds) suitable for CFG sampling.
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    def _tokenize(text: str) -> torch.Tensor:
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        return tokens.input_ids.to(device)

    cond_ids = _tokenize(prompt)
    uncond_ids = _tokenize(negative_prompt)

    cond = text_encoder(cond_ids)[0]
    uncond = text_encoder(uncond_ids)[0]
    return cond, uncond


def _prepare_latents(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """
    Initialize latent noise with stable shape for Stable Diffusion-style models.
    Latent shape is (B, 4, H/8, W/8).
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    h = height // 8
    w = width // 8
    latents = torch.randn((batch_size, 4, h, w), generator=generator, device=device, dtype=dtype)
    return latents


@torch.no_grad()
def sample_latents_cfg(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    cfg: SampleConfig = SampleConfig(),
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Run classifier-free guidance (CFG) sampling and return final latents.

    This is the core "iterative denoise" engine, extracted from the notebook
    and written as reusable software.
    """
    seed_everything(cfg.seed)

    pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    core = DiffusionCore(pipe.scheduler)

    # Determine dtype in a device-aware way.
    # CUDA: float16 for speed; MPS/CPU: float32 for stability/compatibility.
    if device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Text embeddings for CFG
    cond_embeds, uncond_embeds = encode_prompt(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
    )

    # Latent initialization
    latents = _prepare_latents(
        batch_size=1,
        height=cfg.height,
        width=cfg.width,
        device=device,
        dtype=dtype,
        seed=cfg.seed,
    )

    unet = pipe.unet

    with _autocast_if_cuda(device):
        for i, t in enumerate(timesteps):
            # Determine prev_t for the DDPM step
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            is_last_step = (i == len(timesteps) - 1)

            # UNet forward pass: unconditional + conditional
            uncond_out = unet(latents, t, encoder_hidden_states=uncond_embeds)
            cond_out = unet(latents, t, encoder_hidden_states=cond_embeds)

            uncond_pred = uncond_out.sample if hasattr(uncond_out, "sample") else uncond_out
            cond_pred = cond_out.sample if hasattr(cond_out, "sample") else cond_out

            # CFG: combine unconditional and conditional noise predictions
            eps = uncond_pred + cfg.guidance_scale * (cond_pred - uncond_pred)

            # Manual DDPM reverse step (the core diffusion math)
            latents = core.ddpm_step(latents, t, prev_t, eps, add_noise=not is_last_step)

    return latents


@torch.no_grad()
def decode_latents_to_pil(pipe, latents: torch.Tensor) -> Image.Image:
    """
    Decode latents through the VAE and return a PIL image.
    """
    vae = pipe.vae

    # SD convention: scale latents before decoding
    if hasattr(vae, "config") and hasattr(vae.config, "scaling_factor"):
        latents = latents / vae.config.scaling_factor
    else:
        # common fallback value used in SD pipelines
        latents = latents / 0.18215

    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


@torch.no_grad()
def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    cfg: SampleConfig = SampleConfig(),
    device: torch.device = DEVICE,
) -> Image.Image:
    """
    High-level convenience API: text -> image.

    Returns:
        A PIL image generated by CFG sampling.
    """
    latents = sample_latents_cfg(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        cfg=cfg,
        device=device,
    )
    return decode_latents_to_pil(pipe, latents)