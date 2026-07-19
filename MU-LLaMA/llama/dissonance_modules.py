"""Lightweight Dissonance Spectrum encoders and gated residual fusion."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / denominator


class CNNSmallEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        channels = (32, 64, 128)
        self.network = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(channels[-1], hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, spectrum: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        features = self.network(spectrum.unsqueeze(1))
        resized_mask = F.interpolate(
            time_mask[:, None, :].to(features.dtype), size=features.shape[-1], mode="nearest"
        ).squeeze(1)
        pooled_frequency = features.mean(dim=2)
        pooled = _masked_mean(pooled_frequency, resized_mask[:, None, :], dim=-1)
        return self.output(pooled)


class MLPPoolEncoder(nn.Module):
    def __init__(self, frequency_bins: int, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.frequency_bins = int(frequency_bins)
        self.time_summary_bins = 16
        self.network = nn.Sequential(
            nn.Linear(self.frequency_bins * 2 + self.time_summary_bins * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, spectrum: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        if spectrum.shape[1] != self.frequency_bins:
            raise ValueError(
                f"mlp_pool expected {self.frequency_bins} bins, got {spectrum.shape[1]}"
            )
        mask = time_mask[:, None, :].to(spectrum.dtype)
        mean = _masked_mean(spectrum, mask, dim=-1)
        variance = _masked_mean((spectrum - mean.unsqueeze(-1)) ** 2, mask, dim=-1)
        time_stats = []
        for sample, valid_frames in zip(spectrum, time_mask):
            valid_sample = sample[:, valid_frames]
            if valid_sample.shape[-1] == 0:
                valid_sample = sample.new_zeros(sample.shape[0], 1)
            frequency_mean = valid_sample.mean(dim=0)
            frequency_std = valid_sample.std(dim=0, unbiased=False)
            time_stats.append(torch.cat((
                F.adaptive_avg_pool1d(
                    frequency_mean.view(1, 1, -1), self.time_summary_bins
                ).view(-1),
                F.adaptive_avg_pool1d(
                    frequency_std.view(1, 1, -1), self.time_summary_bins
                ).view(-1),
            )))
        time_stats = torch.stack(time_stats)
        stats = torch.cat((mean, variance.clamp_min(0).sqrt(), time_stats), dim=-1)
        return self.network(stats)


def build_ds_encoder(config: Dict, frequency_bins: int) -> nn.Module:
    encoder_type = config.get("type", "cnn_small")
    kwargs = {
        "embedding_dim": int(config.get("embedding_dim", 1024)),
        "hidden_dim": int(config.get("hidden_dim", 256)),
        "dropout": float(config.get("dropout", 0.1)),
    }
    if encoder_type == "cnn_small":
        return CNNSmallEncoder(**kwargs)
    if encoder_type == "mlp_pool":
        return MLPPoolEncoder(frequency_bins=frequency_bins, **kwargs)
    raise ValueError(f"Unknown DS encoder type: {encoder_type}")


class GatedResidualFusion(nn.Module):
    """Low-rank projection and scalar/channel gate, bounded below 5M params."""

    def __init__(
        self,
        base_dim: int,
        ds_dim: int,
        hidden_dim: int = 256,
        gate: str = "scalar",
        gate_bias_init: float = -2.0,
    ):
        super().__init__()
        if gate not in {"scalar", "channel"}:
            raise ValueError("fusion.gate must be 'scalar' or 'channel'")
        self.gate_type = gate
        self.ds_projection = nn.Sequential(
            nn.Linear(ds_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, base_dim, bias=False),
        )
        gate_dim = 1 if gate == "scalar" else base_dim
        self.gate_layer = nn.Sequential(
            nn.Linear(base_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, gate_dim),
        )
        nn.init.constant_(self.gate_layer[-1].bias, float(gate_bias_init))

    def forward(
        self, base: torch.Tensor, ds_embedding: torch.Tensor, valid: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        projected = self.ds_projection(ds_embedding)
        gate = torch.sigmoid(self.gate_layer(torch.cat((base, projected), dim=-1)))
        residual = gate * projected
        if valid is not None:
            residual = residual * valid.to(residual.dtype).view(-1, 1)
        fused = base + residual
        eps = torch.finfo(base.dtype).eps if base.dtype.is_floating_point else 1e-8
        stats = {
            "gate_mean": gate.detach().float().mean(),
            "gate_std": gate.detach().float().std(unbiased=False),
            "ds_embedding_norm": ds_embedding.detach().float().norm(dim=-1).mean(),
            "mert_embedding_norm": base.detach().float().norm(dim=-1).mean(),
            "fused_embedding_norm": fused.detach().float().norm(dim=-1).mean(),
            "ds_residual_ratio": (
                residual.detach().float().norm(dim=-1)
                / base.detach().float().norm(dim=-1).clamp_min(float(eps))
            ).mean(),
        }
        return fused, stats
