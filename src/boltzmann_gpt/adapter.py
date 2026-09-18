"""Belief adapter: MLP from the DBM belief vector to soft-prompt embeddings."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class BeliefAdapter(nn.Module):
    """Maps a belief vector H to K soft-prompt embeddings of size D.

    The MLP is Linear -> ReLU per entry of ``hidden_layers`` followed by a
    final Linear to ``K * D``; the reshaped output passes through a LayerNorm
    over the embedding dimension. This is the only trained component of the
    pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        num_soft_tokens: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.num_soft_tokens = num_soft_tokens
        self.embedding_dim = embedding_dim

        layers: List[nn.Module] = []
        dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h
        layers.append(nn.Linear(dim, num_soft_tokens * embedding_dim))

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, beliefs: torch.Tensor) -> torch.Tensor:
        """(batch, input_dim) -> (batch, num_soft_tokens, embedding_dim)."""
        flat = self.mlp(beliefs)
        soft = flat.view(beliefs.shape[0], self.num_soft_tokens, self.embedding_dim)
        return self.norm(soft)
