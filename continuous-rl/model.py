"""Neural network models for continuous-time policy iteration.

ValueNetwork:  V_θ : ℝ^d → ℝ       (approximates value function)
PolicyNetwork: α_φ : ℝ^d → A ⊂ ℝ^m (approximates feedback policy)
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """MLP approximating V : ℝ^d → ℝ."""

    def __init__(self, state_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d) → (batch, 1)"""
        return self.net(x)


class PolicyNetwork(nn.Module):
    """MLP approximating α : ℝ^d → [a_lo, a_hi] ⊂ ℝ^m.

    A sigmoid output layer maps to the compact action set.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.register_buffer("action_low", action_low.float())
        self.register_buffer("action_high", action_high.float())

        layers: list[nn.Module] = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d) → (batch, m)  clamped to [a_lo, a_hi]."""
        raw = self.net(x)  # (batch, m)
        return self.action_low + (self.action_high - self.action_low) * torch.sigmoid(raw)
