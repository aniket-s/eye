"""The static handshape classifier.

A small residual MLP over 42 normalised features, with an **ArcFace** head.

Why ArcFace rather than plain cross-entropy? Three reasons, and the third is the one
that matters most for this product:

1. It was reported as the single largest accuracy gain in a top-50 solution to
   Google's isolated-sign competition.
2. It produces a normalised embedding whose cosine distance to a class centre is a
   meaningful **open-set** score — so "this is not any letter I know" comes almost for
   free, complementing the trained ``none`` class.
3. It makes **user-defined signs** possible without training. A user records five
   examples, the app averages their embeddings into a new centre, and nearest-centroid
   classification recognises it immediately — entirely in the browser, at zero cost.
   That is the Phase 5 custom-signs feature, and it falls out of this choice.

The network stays deliberately small (~200k parameters). The frame budget is 33 ms, of
which MediaPipe takes 15–25 ms, so the classifier gets under 10 ms.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalise(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project onto the unit hypersphere.

    Written explicitly rather than via ``F.normalize`` because that exports to a graph
    containing ``Expand`` and ``Shape``, which defeats ONNX shape inference and makes
    the model unquantisable. This formulation exports to ReduceSum/Sqrt/Clip/Div —
    numerically identical, and it keeps the int8 path available.
    """
    norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True)).clamp_min(eps)
    return x / norm


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.expand = nn.Linear(width, width * 2)
        self.contract = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.expand(x))
        x = self.dropout(x)
        x = self.contract(x)
        return residual + x


class HandshapeEncoder(nn.Module):
    """Maps normalised landmarks to an L2-normalised embedding."""

    def __init__(
        self,
        feature_length: int = 42,
        width: int = 192,
        blocks: int = 3,
        embedding: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Linear(feature_length, width)
        self.blocks = nn.ModuleList(ResidualBlock(width, dropout) for _ in range(blocks))
        self.norm = nn.LayerNorm(width)
        self.project = nn.Linear(width, embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.project(self.norm(x))
        # Putting every embedding on the unit hypersphere is what makes cosine
        # similarity to a class centre a well-behaved score.
        return l2_normalise(x)


class ArcMarginProduct(nn.Module):
    """ArcFace head: cosine similarity to learned class centres, with an angular margin.

    The margin is applied to the *true* class during training only, forcing each class
    into a tighter cluster than plain cross-entropy would. At inference the head is a
    plain scaled cosine similarity, so export is straightforward.
    """

    def __init__(self, embedding: int, classes: int, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self._cos_m = math.cos(margin)
        self._sin_m = math.sin(margin)
        # Beyond this the margin would push the angle past pi, folding it back and
        # producing a *lower* loss for a worse prediction.
        self._threshold = math.cos(math.pi - margin)
        self._safety = math.sin(math.pi - margin) * margin

    def forward(self, embedding: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(embedding, l2_normalise(self.weight))
        if target is None:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-9))
        penalised = cosine * self._cos_m - sine * self._sin_m
        penalised = torch.where(cosine > self._threshold, penalised, cosine - self._safety)

        one_hot = torch.zeros_like(cosine).scatter_(1, target.view(-1, 1), 1.0)
        return (one_hot * penalised + (1.0 - one_hot) * cosine) * self.scale


class HandshapeClassifier(nn.Module):
    """Encoder plus ArcFace head.

    ``forward`` returns logits. Pass ``target`` during training to apply the margin;
    omit it for evaluation and export so the exported graph has no training-only path.
    """

    def __init__(
        self,
        classes: int,
        feature_length: int = 42,
        width: int = 192,
        blocks: int = 3,
        embedding: int = 128,
        dropout: float = 0.2,
        scale: float = 30.0,
        margin: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = HandshapeEncoder(feature_length, width, blocks, embedding, dropout)
        self.head = ArcMarginProduct(embedding, classes, scale, margin)

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encoder(x), target)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embeddings only — the basis for few-shot custom signs."""
        return self.encoder(x)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
