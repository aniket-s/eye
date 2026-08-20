"""Isolated word-sign classifier.

A sequence of frames in, one of ~250 signs out. Architecture follows the solution that
won Google's isolated sign recognition competition: alternating **1-D convolution
blocks** and **self-attention blocks**, then global pooling.

Two deliberate departures from the fingerspelling model:

* **Not causal.** Isolated recognition classifies a *completed* segment, so the whole
  clip is available and there is no reason to hide the future from the model. Centred
  convolutions and full attention are strictly more informative. (The segmenter that
  decides where a sign ends does run causally — that constraint lives there instead.)
* **Motion features are computed inside the graph.** Position alone cannot distinguish
  signs that share a handshape and differ in movement, and the competition results put
  first and second temporal differences among the cheapest accuracy available. Building
  them into ``forward`` means the browser sends raw normalised frames and cannot
  compute them differently from training — one less place for train/inference skew.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def motion_features(x: torch.Tensor) -> torch.Tensor:
    """Concatenate position with first and second temporal differences.

    ``(batch, frames, features)`` in, ``(batch, frames, features * 3)`` out. Differences
    are zero-padded at the start so the frame count is unchanged.
    """
    first = torch.zeros_like(x)
    first[:, 1:] = x[:, 1:] - x[:, :-1]

    second = torch.zeros_like(x)
    second[:, 2:] = x[:, 2:] - x[:, :-2]

    return torch.cat([x, first, second], dim=-1)


class ConvBlock(nn.Module):
    """Inverted-residual block with a centred depthwise convolution."""

    def __init__(self, channels: int, kernel_size: int = 17, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.BatchNorm1d(channels)
        self.expand = nn.Conv1d(channels, hidden, 1)
        self.depthwise = nn.Conv1d(
            hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden
        )
        self.contract = nn.Conv1d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.silu(self.expand(x))
        x = self.depthwise(x)
        x = self.contract(x)
        return residual + x


class AttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention followed by a feed-forward layer.

    No positional encoding: the convolution blocks either side already encode position
    through their receptive fields, and the competition solutions found adding one
    unnecessary.
    """

    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.norm_feedforward = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, frames, channels)
        normed = self.norm_attention(x)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attended
        return x + self.feedforward(self.norm_feedforward(x))


class IsolatedSignClassifier(nn.Module):
    """Word-level sign classifier over two-handed body-relative features."""

    def __init__(
        self,
        classes: int,
        feature_length: int = 100,
        width: int = 192,
        stages: int = 2,
        conv_per_stage: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Linear(feature_length * 3, width, bias=False)
        self.stem_norm = nn.BatchNorm1d(width)

        self.stages = nn.ModuleList()
        for _ in range(stages):
            self.stages.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(ConvBlock(width) for _ in range(conv_per_stage)),
                        AttentionBlock(width, heads, dropout),
                    ]
                )
            )

        self.project = nn.Linear(width, width * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width * 2, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Motion features inside the graph, so the browser cannot compute them
        # differently from training.
        x = motion_features(x)
        x = self.stem(x)

        # BatchNorm1d over (batch, channels, frames).
        x = self.stem_norm(x.transpose(1, 2)).transpose(1, 2)

        for convolutions, attention in self.stages:
            x = x.transpose(1, 2)
            for block in convolutions:
                x = block(x)
            x = x.transpose(1, 2)
            x = attention(x)

        x = F.silu(self.project(x))
        # Mean over time: a sign is identified by the clip as a whole, not by any one
        # frame, and pooling makes the model indifferent to clip length.
        x = x.mean(dim=1)
        return self.head(self.dropout(x))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
