# src/diffusion.py
"""
Core diffusion mechanics.

This module implements the mathematical operations required for
Denoising Diffusion Probabilistic Models (DDPM), independent of
any model architecture or execution pipeline.

Key equations implemented:
- Forward process: q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
- Reverse step:    x_{t-1} = μ_θ(x_t, t) + σ_t z,  where z ~ N(0, I)

References:
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- https://arxiv.org/abs/2006.11239
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class DiffusionCore:
    """
    Stateless wrapper around a diffusion scheduler.

    This class exposes low-level diffusion operations while remaining
    agnostic to model architecture and sampling strategy.
    """
    scheduler: object

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        """Cumulative product of alphas provided by the scheduler."""
        return self.scheduler.alphas_cumprod

    def alpha_bar(self, t: torch.Tensor | int) -> torch.Tensor:
        """
        Return the cumulative noise scaling factor at timestep t.
        """
        return self.alphas_cumprod[t]

    def forward_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor | int,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion to a clean sample.

        Implements the closed-form forward process:

            x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   where ε ~ N(0, I)

        This allows direct sampling of any timestep without iterating.

        Args:
            x0: Clean input sample.
            t: Diffusion timestep (0 = clean, T = fully noised).
            noise: Optional noise tensor. If omitted, noise is sampled.

        Returns:
            A tuple (x_t, ε) of the noised sample and the noise used.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        a_bar = self.alpha_bar(t)
        while a_bar.ndim < x0.ndim:
            a_bar = a_bar.view(*a_bar.shape, *([1] * (x0.ndim - a_bar.ndim)))

        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
        return xt, noise

    def _match(self, ref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Cast tensor x to match the device and dtype of ref."""
        return x.to(device=ref.device, dtype=ref.dtype)

    def predict_x0_from_eps(self, xt, t, eps):
        """
        Reconstruct a clean sample estimate from noisy input and predicted noise.

        Rearranging the forward equation x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε gives:

            x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t

        This is the "predicted x_0" used in the reverse step.
        """
        a_bar = self._match(xt, self.alpha_bar(t))
        while a_bar.ndim < xt.ndim:
            a_bar = a_bar.view(*a_bar.shape, *([1] * (xt.ndim - a_bar.ndim)))
        return (xt - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar)

    def ddpm_step(self, xt, t, prev_t, eps, add_noise=True):
        """
        Perform a single DDPM reverse diffusion step.

        Implements the posterior mean formula from Ho et al. (2020):

            x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ) + σ_t * z

        where:
            ᾱ_t = cumulative product of α up to step t
            α_t = ᾱ_t / ᾱ_{t-1}  (single-step scaling)
            β_t = 1 - α_t
            σ_t = √β_t  (simplified variance)

        Args:
            xt: Current noisy sample at timestep t.
            t: Current timestep.
            prev_t: Previous timestep in the sampling schedule.
            eps: Predicted noise ε_θ(x_t, t) from the UNet.
            add_noise: Whether to add stochastic noise (False for final step).

        Returns:
            The denoised sample at timestep prev_t.
        """
        a_bar = self._match(xt, self.alpha_bar(t))
        a_bar_prev = self._match(xt, self.alpha_bar(prev_t))

        # Expand dimensions to match latent shape
        while a_bar.ndim < xt.ndim:
            a_bar = a_bar.unsqueeze(-1)
        while a_bar_prev.ndim < xt.ndim:
            a_bar_prev = a_bar_prev.unsqueeze(-1)

        # Compute single-step alpha and beta
        alpha = a_bar / a_bar_prev
        beta = 1.0 - alpha

        # Predict x_0 from x_t and predicted noise
        x0_pred = self.predict_x0_from_eps(xt, t, eps)

        # Clamp x0 prediction to reasonable range for stability
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Posterior mean: weighted combination of x0_pred and xt
        coef_x0 = torch.sqrt(a_bar_prev) * beta / (1.0 - a_bar)
        coef_xt = torch.sqrt(alpha) * (1.0 - a_bar_prev) / (1.0 - a_bar)
        mean = coef_x0 * x0_pred + coef_xt * xt

        if add_noise and prev_t > 0:
            # Posterior variance (simplified)
            variance = beta * (1.0 - a_bar_prev) / (1.0 - a_bar)
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(variance) * noise

        return mean