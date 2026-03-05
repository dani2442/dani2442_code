"""Stochastic LQR — closed-form benchmark for policy iteration.

Dynamics:  dX = (A X + B a) dt + D dW   (additive noise)
Reward:    r(x,a) = -½ (xᵀ Q x + aᵀ R a)
Value:     V(x) = -½ xᵀ P x − c,   c = Tr(DDᵀ P) / (2ρ)
Policy:    a*(x) = −R⁻¹ Bᵀ P x

P solves the discounted Algebraic Riccati Equation (ARE):
    ρ P = Q + Aᵀ P + P A − P B R⁻¹ Bᵀ P

Presets provided: 1-D scalar and 2-D double integrator.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import torch
import matplotlib.pyplot as plt

from policy_iteration import ControlProblem, PolicyIteration, PIConfig
from visualization import plot_convergence, plot_value_comparison_1d, plot_policy_comparison_1d


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

def solve_are(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    D: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve the discounted ARE and return (P, c, K).

    The standard scipy CARE solves:
        Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0
    We need:
        ρ P = Q + Aᵀ P + P A − P B R⁻¹ Bᵀ P
    ⟺  (A − ½ρ I)ᵀ P + P (A − ½ρ I) − P B R⁻¹ Bᵀ P + Q = 0

    So we set  A_tilde = A − ½ρ I  and solve the standard CARE.
    """
    d = A.shape[0]
    A_tilde = A - 0.5 * rho * np.eye(d)
    P = scipy.linalg.solve_continuous_are(A_tilde, B, Q, R)
    P = 0.5 * (P + P.T)  # enforce symmetry

    c = np.trace(D @ D.T @ P) / (2 * rho)
    K = np.linalg.solve(R, B.T @ P)  # R⁻¹ Bᵀ P
    return P, c, K


def analytical_value(x: np.ndarray, P: np.ndarray, c: float) -> np.ndarray:
    """V*(x) = -½ xᵀ P x − c.  x: (..., d) → (...)"""
    return -0.5 * np.einsum("...i,ij,...j->...", x, P, x) - c


def analytical_policy(x: np.ndarray, K: np.ndarray) -> np.ndarray:
    """a*(x) = −K x.  x: (..., d) → (..., m)"""
    return -x @ K.T


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

class StochasticLQR(ControlProblem):
    """General d-dimensional stochastic LQR with additive noise."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        D: np.ndarray,
        a_max: float = 5.0,
        state_clamp: float = 10.0,
    ):
        self.A_np, self.B_np, self.Q_np, self.R_np, self.D_np = A, B, Q, R, D
        self._d = A.shape[0]
        self._m = B.shape[1]
        self._a_max = a_max
        self._state_clamp = state_clamp

        # Pre-convert to tensors (moved to device lazily)
        self._A = torch.from_numpy(A).float()
        self._B = torch.from_numpy(B).float()
        self._Q = torch.from_numpy(Q).float()
        self._R = torch.from_numpy(R).float()
        self._D = torch.from_numpy(D).float()

    @property
    def state_dim(self) -> int:
        return self._d

    @property
    def action_dim(self) -> int:
        return self._m

    @property
    def noise_type(self) -> str:
        return "additive"

    @property
    def state_clamp(self) -> float | None:
        return self._state_clamp

    def _to(self, dev: torch.device):
        """Move constant matrices to given device (lazy)."""
        if self._A.device != dev:
            self._A = self._A.to(dev)
            self._B = self._B.to(dev)
            self._Q = self._Q.to(dev)
            self._R = self._R.to(dev)
            self._D = self._D.to(dev)

    def drift(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        self._to(x.device)
        # (batch, d) = x @ Aᵀ + a @ Bᵀ
        return x @ self._A.T + a @ self._B.T

    def diffusion(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Σ(x,a) = D  (constant, additive).  → (batch, d, m_noise)"""
        self._to(x.device)
        return self._D.unsqueeze(0).expand(x.shape[0], -1, -1)

    def diffusion_sde(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """For additive noise torchsde wants (batch, d, m_noise)."""
        self._to(x.device)
        return self._D.unsqueeze(0).expand(x.shape[0], -1, -1)

    def reward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        self._to(x.device)
        # -½ (xᵀ Q x + aᵀ R a)
        qform_x = (x @ self._Q * x).sum(dim=-1, keepdim=True)
        qform_a = (a @ self._R * a).sum(dim=-1, keepdim=True)
        return -0.5 * (qform_x + qform_a)

    def sample_initial_states(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.randn(n, self._d, device=device) * 2.0

    def action_low(self, device: torch.device) -> torch.Tensor:
        return torch.full((self._m,), -self._a_max, device=device)

    def action_high(self, device: torch.device) -> torch.Tensor:
        return torch.full((self._m,), self._a_max, device=device)


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def preset_1d(sigma: float = 0.3) -> tuple[StochasticLQR, dict]:
    """1-D scalar LQR:  dX = (α x + β a) dt + σ dW."""
    alpha, beta = -0.5, 1.0
    q, r = 1.0, 0.1
    A = np.array([[alpha]])
    B = np.array([[beta]])
    Q = np.array([[q]])
    R = np.array([[r]])
    D = np.array([[sigma]])
    return StochasticLQR(A, B, Q, R, D, a_max=5.0), dict(A=A, B=B, Q=Q, R=R, D=D)


def preset_2d(sigma: float = 0.2) -> tuple[StochasticLQR, dict]:
    """2-D double integrator:  ẋ₁ = x₂,  ẋ₂ = a + noise."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.diag([1.0, 1.0])
    R = np.array([[0.1]])
    D = np.array([[0.0, 0.0], [0.0, sigma]])
    return StochasticLQR(A, B, Q, R, D, a_max=5.0), dict(A=A, B=B, Q=Q, R=R, D=D)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Choose preset ──────────────────────────────────────────────────
    rho = 0.1
    problem, params = preset_1d(sigma=0.3)

    # ── Analytical solution ────────────────────────────────────────────
    P, c, K = solve_are(**params, rho=rho)
    print(f"\n── Analytical solution ──")
    print(f"P = {P}")
    print(f"c = {c:.6f}")
    print(f"K = {K}  (a* = -K x)")

    # Evaluate on a grid within the feasible action range
    xs = np.linspace(-2, 2, 200).reshape(-1, 1)
    V_exact = analytical_value(xs, P, c)
    a_exact = analytical_policy(xs, K)

    # ── Policy iteration ──────────────────────────────────────────────
    cfg = PIConfig(
        rho=rho,
        T=8.0,
        dt=0.05,
        n_trajectories=128,
        n_eval_steps=200,
        n_improve_steps=30,
        lr_value=3e-3,
        lr_policy=3e-4,
        hidden_dim=64,
        n_layers=2,
        n_collocation=512,
        n_outer=25,
        device=device,
    )
    solver = PolicyIteration(problem, cfg)

    print("\n── Running policy iteration ──")
    history = solver.solve(verbose=True)

    # ── Compare learned vs analytical ─────────────────────────────────
    xs_t = torch.from_numpy(xs).float().to(device)
    with torch.no_grad():
        V_learned = solver.V_net(xs_t).cpu().numpy().flatten()
        a_learned = solver.policy_net(xs_t).cpu().numpy().flatten()

    # ── Monte-Carlo validation ────────────────────────────────────────
    x0 = torch.tensor([1.0])
    V_pred, V_mc = solver.mc_validate(x0, n_sims=2048, T=30.0)
    V_true = analytical_value(np.array([[1.0]]), P, c).item()
    print(f"\n── Monte-Carlo validation at x₀ = 1.0 ──")
    print(f"  V_learned(x₀) = {V_pred:.4f}")
    print(f"  V_MC(x₀)      = {V_mc:.4f}")
    print(f"  V_exact(x₀)   = {V_true:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────
    plot_convergence(history)
    plot_value_comparison_1d(
        xs.flatten(), V_exact.flatten(), V_learned,
        title="Stochastic LQR — Value Function",
    )
    plot_policy_comparison_1d(
        xs.flatten(), a_exact.flatten(), a_learned,
        title="Stochastic LQR — Policy",
    )


if __name__ == "__main__":
    main()
