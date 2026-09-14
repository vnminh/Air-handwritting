from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def branch_dimensions(representation: str, fourier_scales: int) -> dict[str, int]:
    dimensions: dict[str, int] = {}
    if representation in {"xy", "xy_delta", "xy_dynamics", "xy_fourier", "all"}:
        dimensions["spatial"] = 2
    if representation == "xy_delta":
        dimensions["dynamic"] = 2
    elif representation in {"dynamics", "xy_dynamics", "dynamics_fourier", "all"}:
        dimensions["dynamic"] = 7
    if representation in {"fourier", "xy_fourier", "dynamics_fourier", "all"}:
        dimensions["fourier"] = 4 * fourier_scales
    if not dimensions:
        raise ValueError(f"Unknown representation: {representation}")
    return dimensions


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_length: int = 1024) -> None:
        super().__init__()
        position = torch.arange(max_length).float().unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[:, : inputs.shape[1]]


class RelativeSelfAttention(nn.Module):
    """Multi-head self-attention with a learned clipped relative-position bias."""

    def __init__(self, d_model: int, heads: int, dropout: float, max_distance: int = 64) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.max_distance = max_distance
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.relative_bias = nn.Embedding(2 * max_distance + 1, heads)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, width = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        positions = torch.arange(length, device=inputs.device)
        distance = positions[None, :] - positions[:, None]
        distance = distance.clamp(-self.max_distance, self.max_distance) + self.max_distance
        bias = self.relative_bias(distance).permute(2, 0, 1)
        logits = logits + bias.unsqueeze(0)
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attention = self.dropout(F.softmax(logits, dim=-1))
        result = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, width)
        return self.output(result)


class BranchFusion(nn.Module):
    def __init__(
        self,
        dimensions: dict[str, int],
        d_model: int,
        strategy: str,
        dropout: float,
        branch_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= branch_dropout < 1.0:
            raise ValueError("branch_dropout must be in [0, 1)")
        self.names = list(dimensions)
        self.strategy = strategy
        self.branch_dropout = branch_dropout
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(width, d_model), nn.LayerNorm(d_model), nn.SiLU())
                for name, width in dimensions.items()
            }
        )
        count = len(self.names)
        self.concat_projection = nn.Linear(count * d_model, d_model) if count > 1 and strategy == "concat" else None
        self.static_logits = nn.Parameter(torch.zeros(count)) if count > 1 and strategy == "static" else None
        self.gate = nn.Linear(count * d_model, count) if count > 1 and strategy == "gated" else None
        self.dropout = nn.Dropout(dropout)

    def _branch_mask(self, stacked: torch.Tensor) -> torch.Tensor | None:
        if not self.training or self.branch_dropout <= 0.0 or stacked.shape[2] == 1:
            return None
        batch, _, count, _ = stacked.shape
        keep = torch.rand(batch, count, device=stacked.device) >= self.branch_dropout
        empty = ~keep.any(dim=1)
        if empty.any():
            fallback = torch.randint(count, (int(empty.sum()),), device=stacked.device)
            keep[empty] = False
            keep[empty, fallback] = True
        return keep[:, None, :]

    def forward(self, branches: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedded = [self.embeddings[name](branches[name]) for name in self.names]
        if len(embedded) == 1:
            return self.dropout(embedded[0]), None
        stacked = torch.stack(embedded, dim=2)
        branch_mask = self._branch_mask(stacked)
        if branch_mask is not None:
            stacked = stacked * branch_mask.unsqueeze(-1)
        concatenated = stacked.flatten(start_dim=2)
        weights = None
        if self.strategy == "concat":
            fused = self.concat_projection(concatenated)
        elif self.strategy == "sum":
            if branch_mask is None:
                fused = stacked.mean(dim=2)
            else:
                fused = stacked.sum(dim=2) / branch_mask.sum(dim=2).clamp_min(1).unsqueeze(-1)
        elif self.strategy == "static":
            logits = self.static_logits[None, None, :].expand(
                stacked.shape[0], stacked.shape[1], -1
            )
            if branch_mask is not None:
                logits = logits.masked_fill(~branch_mask, torch.finfo(logits.dtype).min)
            weights = F.softmax(logits, dim=-1)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        elif self.strategy == "gated":
            logits = self.gate(concatenated)
            if branch_mask is not None:
                logits = logits.masked_fill(~branch_mask, torch.finfo(logits.dtype).min)
            weights = F.softmax(logits, dim=-1)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")
        return self.dropout(fused), weights

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.norm(inputs).transpose(1, 2)
        values = F.glu(self.pointwise_in(values), dim=1)
        values = F.silu(self.batch_norm(self.depthwise(values)))
        return self.dropout(self.pointwise_out(values).transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        d_ff: int,
        dropout: float,
        kernel_size: int,
        relative: bool,
    ) -> None:
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = (
            RelativeSelfAttention(d_model, heads, dropout)
            if relative
            else nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        )
        self.relative = relative
        self.attention_dropout = nn.Dropout(dropout)
        self.convolution = ConformerConvolution(d_model, kernel_size, dropout)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        values = inputs + 0.5 * self.ff1(inputs)
        normed = self.attention_norm(values)
        if self.relative:
            attended = self.attention(normed, padding_mask)
        else:
            attended, _ = self.attention(normed, normed, normed, key_padding_mask=padding_mask, need_weights=False)
        values = values + self.attention_dropout(attended)
        values = values + self.convolution(values)
        values = values + 0.5 * self.ff2(values)
        return self.final_norm(values)


class TemporalBlock(nn.Module):
    def __init__(self, d_model: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.conv1 = nn.Conv1d(d_model, d_model, 5, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, 5, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.dropout(F.silu(self.norm1(self.conv1(inputs))))
        values = self.dropout(F.silu(self.norm2(self.conv2(values))))
        return inputs + values


class RecognitionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        representation: str = "all",
        fusion: str = "gated",
        backbone: str = "conformer",
        positional: str = "relative",
        fourier_scales: int = 4,
        d_model: int = 128,
        layers: int = 4,
        heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        kernel_size: int = 15,
        branch_dropout: float = 0.0,
        time_mask_probability: float = 0.0,
        time_mask_width: int = 0,
    ) -> None:
        super().__init__()
        if not 0.0 <= time_mask_probability <= 1.0:
            raise ValueError("time_mask_probability must be in [0, 1]")
        if time_mask_width < 0:
            raise ValueError("time_mask_width must be non-negative")
        self.backbone_name = backbone
        self.positional_name = positional
        self.time_mask_probability = time_mask_probability
        self.time_mask_width = time_mask_width
        self.fusion = BranchFusion(
            branch_dimensions(representation, fourier_scales),
            d_model,
            fusion,
            dropout,
            branch_dropout,
        )
        self.absolute_position = SinusoidalPosition(d_model) if positional == "absolute" else None

        if backbone == "conformer":
            self.backbone = nn.ModuleList(
                [
                    ConformerBlock(
                        d_model, heads, d_ff, dropout, kernel_size, relative=positional == "relative"
                    )
                    for _ in range(layers)
                ]
            )
        elif backbone == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, batch_first=True, norm_first=True, activation="gelu"
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
        elif backbone in {"bilstm", "gru"}:
            recurrent = nn.LSTM if backbone == "bilstm" else nn.GRU
            self.backbone = recurrent(
                d_model, d_model // 2, num_layers=layers, dropout=dropout if layers > 1 else 0.0,
                batch_first=True, bidirectional=True,
            )
        elif backbone == "tcn":
            self.backbone = nn.ModuleList([TemporalBlock(d_model, 2**index, dropout) for index in range(layers)])
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.output = nn.Linear(d_model, num_classes)

    def forward(
        self, branches: dict[str, torch.Tensor], lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        values, fusion_weights = self.fusion(branches)
        if self.training and self.time_mask_probability > 0.0 and self.time_mask_width > 0:
            values = values.clone()
            for batch_index, valid_length in enumerate(lengths.tolist()):
                if valid_length > 1 and torch.rand((), device=values.device) < self.time_mask_probability:
                    width = min(self.time_mask_width, valid_length - 1)
                    start = int(torch.randint(valid_length - width + 1, (), device=values.device))
                    values[batch_index, start : start + width] = 0.0
        if self.absolute_position is not None:
            values = self.absolute_position(values)
        positions = torch.arange(values.shape[1], device=values.device)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)

        if self.backbone_name == "conformer":
            for block in self.backbone:
                values = block(values, padding_mask)
        elif self.backbone_name == "transformer":
            values = self.backbone(values, src_key_padding_mask=padding_mask)
        elif self.backbone_name in {"bilstm", "gru"}:
            values, _ = self.backbone(values)
        elif self.backbone_name == "tcn":
            values = values.transpose(1, 2)
            for block in self.backbone:
                values = block(values)
            values = values.transpose(1, 2)

        log_probs = F.log_softmax(self.output(values), dim=-1).transpose(0, 1)
        return log_probs, lengths, fusion_weights

