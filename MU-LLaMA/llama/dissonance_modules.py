"""Lightweight global/temporal Dissonance Spectrum encoders and fusion."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / denominator


def _resize_time_mask(time_mask: torch.Tensor, size: int) -> torch.Tensor:
    """Resize binary masks in FP32 for PyTorch versions without BF16 interpolation."""
    return F.interpolate(
        time_mask[:, None, :].to(dtype=torch.float32), size=size, mode="nearest"
    ).squeeze(1).ge(0.5)


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

    def extract_feature_map(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Return the ordered CNN map without changing the legacy forward path."""
        return self.network(spectrum.unsqueeze(1))

    def pool_feature_map(self, features: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        resized_mask = _resize_time_mask(time_mask, features.shape[-1])
        pooled_frequency = features.mean(dim=2)
        pooled = _masked_mean(pooled_frequency, resized_mask[:, None, :], dim=-1)
        return self.output(pooled)

    def forward(self, spectrum: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        return self.pool_feature_map(self.extract_feature_map(spectrum), time_mask)


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


def sinusoidal_positions(length: int, dimension: int, device, dtype) -> torch.Tensor:
    """Parameter-free sinusoidal positions in chronological order."""
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scales = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, dimension))
    )
    encoding = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * scales)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(positions * scales[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class DSTemporalEncoder(nn.Module):
    """Turn the CNN feature map into masked, ordered DS tokens."""

    def __init__(
        self,
        input_dim: int = 128,
        token_dim: int = 256,
        conv_kernel: int = 5,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",
    ):
        super().__init__()
        if conv_kernel < 1 or conv_kernel % 2 == 0:
            raise ValueError("temporal.conv_kernel must be a positive odd integer")
        if positional_encoding not in {"sinusoidal", "none"}:
            raise ValueError("temporal.positional_encoding must be sinusoidal or none")
        self.token_dim = int(token_dim)
        self.positional_encoding = positional_encoding
        self.input_projection = nn.Linear(int(input_dim), self.token_dim)
        self.depthwise = nn.Conv1d(
            self.token_dim, self.token_dim, int(conv_kernel),
            padding=int(conv_kernel) // 2, groups=self.token_dim,
        )
        self.pointwise = nn.Conv1d(self.token_dim, self.token_dim, 1)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.token_dim)

    def forward(
        self, feature_map: torch.Tensor, time_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if feature_map.ndim != 4:
            raise ValueError("DS temporal input must have shape [B,C,F,T]")
        resized_mask = _resize_time_mask(time_mask, feature_map.shape[-1])
        # Frequency is summarized, but time is deliberately retained and ordered.
        tokens = feature_map.mean(dim=2).transpose(1, 2)
        tokens = self.input_projection(tokens)
        tokens = tokens * resized_mask.unsqueeze(-1).to(tokens.dtype)
        if self.positional_encoding == "sinusoidal":
            tokens = tokens + sinusoidal_positions(
                tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype
            ).unsqueeze(0) * resized_mask.unsqueeze(-1).to(tokens.dtype)
        convolution = self.pointwise(self.depthwise(tokens.transpose(1, 2))).transpose(1, 2)
        tokens = self.norm(tokens + self.dropout(F.gelu(convolution)))
        tokens = tokens * resized_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, resized_mask


def build_ds_temporal_encoder(config: Dict, input_dim: int = 128) -> Optional[DSTemporalEncoder]:
    if not bool(config.get("enabled", False)):
        return None
    return DSTemporalEncoder(
        input_dim=input_dim,
        token_dim=int(config.get("token_dim", 256)),
        conv_kernel=int(config.get("conv_kernel", 5)),
        dropout=float(config.get("dropout", 0.1)),
        positional_encoding=config.get("positional_encoding", "sinusoidal"),
    )


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


class TemporalGatedAttentionFusion(nn.Module):
    """Single-query masked attention followed by a zero-init gated residual."""

    def __init__(
        self,
        base_dim: int,
        token_dim: int = 256,
        attention_dim: int = 256,
        num_heads: int = 1,
        gate: str = "scalar",
        gate_bias_init: float = -3.0,
        output_zero_init: bool = True,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if int(num_heads) != 1:
            raise ValueError("Stage 2 temporal fusion currently requires num_heads=1")
        if gate not in {"scalar", "learned_scalar"}:
            raise ValueError("Stage 2 temporal fusion gate must be scalar or learned_scalar")
        self.gate_type = gate
        self.attention_dim = int(attention_dim)
        self.query = nn.Linear(int(base_dim), self.attention_dim, bias=False)
        self.key = nn.Linear(int(token_dim), self.attention_dim, bias=False)
        self.value = nn.Linear(int(token_dim), self.attention_dim, bias=False)
        self.context_projection = nn.Linear(self.attention_dim, int(base_dim))
        if gate == "learned_scalar":
            self.gate_logit = nn.Parameter(torch.tensor(float(gate_bias_init)))
            self.gate_layer = None
        else:
            self.register_parameter("gate_logit", None)
            self.gate_layer = nn.Sequential(
                nn.Linear(int(base_dim) * 2, int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), 1),
            )
            nn.init.constant_(self.gate_layer[-1].bias, float(gate_bias_init))
        if output_zero_init:
            nn.init.zeros_(self.context_projection.weight)
            nn.init.zeros_(self.context_projection.bias)

    def forward(
        self,
        base: torch.Tensor,
        temporal_tokens: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if base.ndim != 2 or temporal_tokens.ndim != 3 or temporal_mask.ndim != 2:
            raise ValueError("temporal fusion expects base [B,D], tokens [B,T,D], mask [B,T]")
        valid = temporal_mask.any(dim=-1)
        safe_mask = temporal_mask.bool().clone()
        if safe_mask.shape[1] == 0:
            raise ValueError("temporal fusion requires at least one padded token slot")
        safe_mask[~valid, 0] = True
        query = self.query(base).unsqueeze(1)
        key = self.key(temporal_tokens)
        value = self.value(temporal_tokens)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.attention_dim)
        scores = scores.masked_fill(~safe_mask[:, None, :], float("-inf"))
        attention = torch.softmax(scores.float(), dim=-1).to(value.dtype)
        context = torch.matmul(attention, value).squeeze(1)
        projected = self.context_projection(context)
        if self.gate_type == "learned_scalar":
            gate = torch.sigmoid(self.gate_logit).to(base.dtype).expand(base.shape[0], 1)
        else:
            gate = torch.sigmoid(self.gate_layer(torch.cat((base, projected), dim=-1)))
        residual = gate * projected * valid.to(base.dtype).unsqueeze(1)
        fused = base + residual
        entropy = -(attention.float().clamp_min(1e-12).log() * attention.float()).sum(dim=-1)
        eps = torch.finfo(base.dtype).eps if base.dtype.is_floating_point else 1e-8
        stats = {
            "attention_entropy": (
                entropy[valid].mean().detach() if valid.any() else entropy.new_zeros(())
            ),
            "gate_mean": gate.detach().float().mean(),
            "gate_std": gate.detach().float().std(unbiased=False),
            "ds_embedding_norm": context.detach().float().norm(dim=-1).mean(),
            "mert_embedding_norm": base.detach().float().norm(dim=-1).mean(),
            "fused_embedding_norm": fused.detach().float().norm(dim=-1).mean(),
            "ds_residual_ratio": (
                residual.detach().float().norm(dim=-1)
                / base.detach().float().norm(dim=-1).clamp_min(float(eps))
            ).mean(),
        }
        return fused, stats
