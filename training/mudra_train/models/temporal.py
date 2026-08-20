"""Temporal encoder with a CTC head, for continuous fingerspelling.

Architecture follows the design that won Google's isolated-sign competition and took
second in the fingerspelling one: a stack of **causal depthwise-separable 1-D
convolutions** with channel attention, rather than a transformer.

Three reasons that shape is right here:

1. **It fits the budget.** After MediaPipe takes 15–25 ms of a 33 ms frame, the
   classifier has under 10 ms. A 5M-parameter transformer over 60 frames costs
   50–120 ms; a ~1M-parameter convolutional stack costs ~10 ms.
2. **Causal convolutions stream.** Each output depends only on past frames, so the
   same weights work offline and live. Bidirectional attention does not.
3. **A wide kernel sees whole letters.** Kernel 17 over a 60-frame window gives each
   layer roughly half a second of context, which is the timescale of a fingerspelled
   letter.

CTC over ``blank + alphabet`` removes segmentation entirely — including J and Z, which
are motion-defined and therefore ordinary labels to a model that sees motion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: CTC blank index. Must match ``BLANK_INDEX`` in packages/core/src/decode/ctc.ts.
BLANK_INDEX = 0


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise 1-D convolution that never reads the future.

    Padding is applied entirely on the left, so output frame *t* depends only on
    inputs up to *t*. This is what makes the same weights usable for streaming.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,  # depthwise: one filter per channel
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, frames)
        return self.conv(F.pad(x, (self.padding, 0)))


class CausalChannelAttention(nn.Module):
    """Per-channel gating computed from **past frames only**.

    Standard ECA pools across the whole sequence, which makes the gate depend on the
    future. That is invisible offline and wrong for streaming: the model would score
    one way in evaluation and behave differently on a live camera, because at frame *t*
    a browser simply does not have frames *t+1* onward.

    A cumulative mean gives each frame a running average over everything seen so far,
    so the gate is causal while still summarising the sequence. ``test_encoder_is_causal``
    is what caught the non-causal version.
    """

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, frames)
        batch, channels, frames = x.shape

        counts = torch.arange(1, frames + 1, device=x.device, dtype=x.dtype)
        running_mean = x.cumsum(dim=-1) / counts  # (batch, channels, frames)

        # One gate per frame, from that frame's running channel statistics.
        per_frame = running_mean.permute(0, 2, 1).reshape(batch * frames, 1, channels)
        weights = torch.sigmoid(self.conv(per_frame))
        weights = weights.reshape(batch, frames, channels).permute(0, 2, 1)

        return x * weights


class Conv1dBlock(nn.Module):
    """Inverted-residual block: expand, causal depthwise conv, gate, contract."""

    def __init__(self, channels: int, kernel_size: int = 17, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.BatchNorm1d(channels)
        self.expand = nn.Conv1d(channels, hidden, 1)
        self.depthwise = CausalDepthwiseConv1d(hidden, kernel_size)
        self.attention = CausalChannelAttention()
        self.contract = nn.Conv1d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.silu(self.expand(x))
        x = self.depthwise(x)
        x = self.attention(x)
        x = self.contract(x)
        return residual + x


class FingerspellCtc(nn.Module):
    """Causal convolutional encoder with a CTC head.

    Input ``(batch, frames, features)``; output ``(batch, frames, classes)`` log
    probabilities, with blank at index 0.
    """

    def __init__(
        self,
        classes: int,
        feature_length: int = 42,
        width: int = 128,
        blocks: int = 6,
        kernel_size: int = 17,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.stem = nn.Conv1d(feature_length, width, 1)
        self.blocks = nn.ModuleList(Conv1dBlock(width, kernel_size) for _ in range(blocks))
        self.norm = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Conv1d(width, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, frames, features) -> (batch, features, frames) for Conv1d
        x = x.transpose(1, 2)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(F.silu(self.norm(x)))
        x = self.head(x)
        # Back to (batch, frames, classes), as log probabilities for CTC.
        return F.log_softmax(x.transpose(1, 2), dim=-1)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ctc_loss(
    log_probabilities: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """CTC loss over ``(batch, frames, classes)`` log probabilities.

    ``nn.CTCLoss`` expects ``(frames, batch, classes)``, hence the transpose. ``zero_infinity``
    guards the case where a target is longer than the input can encode, which produces
    an infinite loss and would otherwise poison the gradients for the whole batch.
    """
    return F.ctc_loss(
        log_probabilities.transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=BLANK_INDEX,
        reduction="mean",
        zero_infinity=True,
    )


def greedy_decode(log_probabilities: torch.Tensor) -> list[list[int]]:
    """Greedy CTC decode: argmax per frame, collapse repeats, strip blanks.

    Mirrors ``ctcGreedyDecode`` in packages/core/src/decode/ctc.ts so that offline
    evaluation measures the same decoder the browser runs.
    """
    predictions = log_probabilities.argmax(dim=-1).cpu().numpy()
    decoded: list[list[int]] = []

    for sequence in predictions:
        collapsed: list[int] = []
        previous = -1
        for index in sequence:
            index = int(index)
            if index != previous and index != BLANK_INDEX:
                collapsed.append(index)
            previous = index
        decoded.append(collapsed)

    return decoded
