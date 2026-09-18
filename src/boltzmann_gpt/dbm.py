"""Deep Boltzmann Machine over binary attribute features (inference only)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DeepBoltzmannMachine(nn.Module):
    """Stack of bipartite binary layers ``[n_visible, J_1, ..., J_L]``.

    Only the operations needed at inference time are implemented: mean-field
    inference, the mean-field energy score, and Gibbs sampling. Training
    (layer-wise pretraining + PCD joint fine-tuning) is not included.
    """

    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least [n_visible, J_1]")

        self.layer_sizes = list(layer_sizes)
        self.n_layers = len(layer_sizes) - 1  # number of hidden layers

        # weights[l] connects layer l and layer l+1; biases[l] belongs to layer l
        # (biases[0] is the visible bias). This layout matches the checkpoint keys.
        self.weights = nn.ParameterList(
            nn.Parameter(torch.zeros(layer_sizes[i], layer_sizes[i + 1]))
            for i in range(self.n_layers)
        )
        self.biases = nn.ParameterList(
            nn.Parameter(torch.zeros(size)) for size in layer_sizes
        )

    @property
    def n_visible(self) -> int:
        return self.layer_sizes[0]

    @property
    def belief_dim(self) -> int:
        """Dimension of the concatenated belief vector H."""
        return sum(self.layer_sizes[1:])

    @torch.no_grad()
    def mean_field_inference(self, v: torch.Tensor, n_iter: int = 10) -> List[torch.Tensor]:
        """Run the fully factorized mean-field fixed point and return the means.

        Each hidden layer receives a bottom-up and (except the top layer) a
        top-down contribution, so the fixed point is iterated rather than
        computed in one pass.
        """
        batch_size = v.shape[0]
        mu = [
            torch.full(
                (batch_size, self.layer_sizes[i + 1]),
                0.5,
                device=v.device,
                dtype=v.dtype,
            )
            for i in range(self.n_layers)
        ]

        for _ in range(n_iter):
            bottom_up = torch.matmul(v, self.weights[0]) + self.biases[1]
            if self.n_layers > 1:
                mu[0] = torch.sigmoid(bottom_up + torch.matmul(mu[1], self.weights[1].t()))
            else:
                mu[0] = torch.sigmoid(bottom_up)

            for i in range(1, self.n_layers - 1):
                bottom_up = torch.matmul(mu[i - 1], self.weights[i]) + self.biases[i + 1]
                top_down = torch.matmul(mu[i + 1], self.weights[i + 1].t())
                mu[i] = torch.sigmoid(bottom_up + top_down)

            if self.n_layers > 1:
                bottom_up = torch.matmul(mu[-2], self.weights[-1]) + self.biases[-1]
                mu[-1] = torch.sigmoid(bottom_up)

        return mu

    @torch.no_grad()
    def energy(self, v: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
        """Mean-field energy score ``F~(v)``, without the entropy term.

        Every hidden unit is scored against its full mean-field input
        (bottom-up, top-down and bias), so couplings between hidden layers
        enter with weight two and the score is not exactly the expected energy
        under the mean field. It reproduces ``free_energy()`` of the reference
        implementation, which is the score used for both PCD training and
        evaluation, so values are comparable with the published ones. Lower is
        more coherent. Returns shape ``(batch,)``.
        """
        mu = self.mean_field_inference(v, n_iter)

        visible_term = torch.matmul(v, self.biases[0])
        hidden_term = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)

        wx_b = torch.matmul(v, self.weights[0]) + self.biases[1]
        if self.n_layers > 1:
            wx_b = wx_b + torch.matmul(mu[1], self.weights[1].t())
        hidden_term = hidden_term + torch.sum(mu[0] * wx_b, dim=1)

        for i in range(1, self.n_layers - 1):
            wx_b = torch.matmul(mu[i - 1], self.weights[i]) + self.biases[i + 1]
            wx_b = wx_b + torch.matmul(mu[i + 1], self.weights[i + 1].t())
            hidden_term = hidden_term + torch.sum(mu[i] * wx_b, dim=1)

        if self.n_layers > 1:
            wx_b = torch.matmul(mu[-2], self.weights[-1]) + self.biases[-1]
            hidden_term = hidden_term + torch.sum(mu[-1] * wx_b, dim=1)

        return -visible_term - hidden_term

    @torch.no_grad()
    def beliefs(self, v: torch.Tensor, n_iter: int = 10, use_top_layer: bool = False) -> torch.Tensor:
        """Belief vector fed to the adapter: the converged mean-field means,
        concatenated across layers (or the top layer only)."""
        mu = self.mean_field_inference(v, n_iter)
        return mu[-1] if use_top_layer else torch.cat(mu, dim=1)

    @torch.no_grad()
    def sample_v_given_h(self, h1: torch.Tensor) -> torch.Tensor:
        """Sample the visible layer from the first hidden layer."""
        p = torch.sigmoid(torch.matmul(h1, self.weights[0].t()) + self.biases[0])
        return torch.bernoulli(p)

    @torch.no_grad()
    def gibbs_sampling(self, v: torch.Tensor, k_steps: int = 1, n_iter: int = 10) -> torch.Tensor:
        """Alternate v -> h -> v for ``k_steps`` and return the visible sample."""
        for _ in range(k_steps):
            mu = self.mean_field_inference(v, n_iter)
            h1 = torch.bernoulli(mu[0])
            v = self.sample_v_given_h(h1)
        return v
