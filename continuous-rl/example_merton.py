"""Merton portfolio / consumption — closed-form benchmark for policy iteration.

State (wealth):
    dX = (r_f + π(μ−r_f) − c_rate) X dt + π σ X dW

Controls:  a = (π, c_rate)  where
    π      ∈ [π_lo, π_hi]    risky-asset fraction
    c_rate ∈ [k_lo, k_hi]    consumption-to-wealth ratio  (c = c_rate · X)

Using c_rate = c/X is the standard finance parameterisation.  It keeps
the SDE purely multiplicative and prevents the catastrophic wealth
depletion that occurs when the neural net tries large absolute c.

Reward (CRRA utility, γ ≠ 1):
    r(x, a) = reward_scale · (c_rate · x)^{1−γ} / (1−γ)

``reward_scale'' rescales V so the neural net targets are O(1).
The optimal policy is **invariant** to this scaling.

Closed-form:
    π*  = (μ − r_f) / (γ σ²)                (constant)
    c_rate* = k = (ρ − (1−γ)M) / γ          (constant)
    M   = r_f + (μ − r_f)² / (2 γ σ²)
    A   = (γ / (ρ − (1−γ)M))^γ
    V*(x) = reward_scale · A/(1−γ) · x^{1−γ}
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from policy_iteration import ControlProblem, PolicyIteration, PIConfig
from visualization import plot_convergence, plot_value_comparison_1d


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

def analytical_merton(
    r_f: float, mu: float, sigma: float, gamma: float, rho: float
) -> dict:
    """Return closed-form optimal controls and value parameters."""
    pi_star = (mu - r_f) / (gamma * sigma ** 2)
    M = r_f + (mu - r_f) ** 2 / (2 * gamma * sigma ** 2)
    denom = rho - (1 - gamma) * M
    assert denom > 0, (
        f"Infeasible: ρ − (1−γ)M = {denom:.4f} ≤ 0.  "
        f"Increase ρ or γ, or decrease μ."
    )
    k = denom / gamma          # optimal c_rate
    A = (gamma / denom) ** gamma
    return dict(pi_star=pi_star, M=M, k=k, A=A, denom=denom)


def analytical_value(
    x: np.ndarray, A: float, gamma: float, reward_scale: float = 1.0
) -> np.ndarray:
    """V*(x) = reward_scale · A/(1−γ) · x^{1−γ}."""
    return reward_scale * A / (1 - gamma) * np.power(x, 1 - gamma)


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

class MertonProblem(ControlProblem):
    """Merton portfolio / consumption with CRRA utility.

    Action vector:  a = (π, c_rate)
        π      – fraction invested in the risky asset
        c_rate – consumption-to-wealth ratio  (c = c_rate · x)

    The multiplicative parameterisation avoids numerical catastrophes.
    """

    def __init__(
        self,
        r_f: float = 0.03,
        mu: float = 0.08,
        sigma: float = 0.20,
        gamma: float = 2.0,
        rho: float = 0.05,
        pi_bounds: tuple[float, float] = (0.0, 1.5),
        crate_bounds: tuple[float, float] = (0.005, 0.20),
        x_range: tuple[float, float] = (0.5, 5.0),
        reward_scale: float = 1.0,
    ):
        self.r_f = r_f
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.rho = rho
        self.pi_lo, self.pi_hi = pi_bounds
        self.cr_lo, self.cr_hi = crate_bounds
        self.x_lo, self.x_hi = x_range
        self.reward_scale = reward_scale

    # ── ControlProblem interface ──────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def action_dim(self) -> int:
        return 2   # (π, c_rate)

    @property
    def noise_type(self) -> str:
        return "diagonal"

    @property
    def state_clamp(self) -> float | None:
        return self.x_hi * 4

    @property
    def state_clamp_lo(self) -> float:
        return 0.01

    def drift(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """f(x, a) = (r_f + π(μ−r_f) − c_rate) · x."""
        pi = a[:, 0:1]
        cr = a[:, 1:2]
        return (self.r_f + pi * (self.mu - self.r_f) - cr) * x

    def diffusion(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Σ(x, a) → (batch, 1, 1) for the generator."""
        pi = a[:, 0:1]
        return (pi * self.sigma * x).unsqueeze(-1)

    def diffusion_sde(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Diagonal noise for torchsde → (batch, 1)."""
        pi = a[:, 0:1]
        return pi * self.sigma * x

    def reward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """reward_scale · (c_rate · x)^{1−γ} / (1−γ)."""
        cr = a[:, 1:2]
        c = (cr * x).clamp(min=1e-8)
        return self.reward_scale * c.pow(1 - self.gamma) / (1 - self.gamma)

    def sample_initial_states(self, n: int, device: torch.device) -> torch.Tensor:
        """Log-uniform in [x_lo, x_hi]."""
        log_lo = np.log(self.x_lo)
        log_hi = np.log(self.x_hi)
        log_x = torch.rand(n, 1, device=device) * (log_hi - log_lo) + log_lo
        return log_x.exp()

    def action_low(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([self.pi_lo, self.cr_lo], device=device)

    def action_high(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([self.pi_hi, self.cr_hi], device=device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Parameters ─────────────────────────────────────────────────────
    r_f, mu, sigma, gamma, rho = 0.03, 0.08, 0.20, 2.0, 0.05
    reward_scale = 0.01          # makes V* ≈ O(1-10) → easier for a NN

    # ── Analytical solution ────────────────────────────────────────────
    sol = analytical_merton(r_f, mu, sigma, gamma, rho)
    print("\n── Analytical (Merton) ──")
    print(f"  π*       = {sol['pi_star']:.4f}")
    print(f"  k (c_rate*) = {sol['k']:.4f}")
    print(f"  A        = {sol['A']:.6f}")
    print(f"  V*(1)    = {analytical_value(np.array([1.0]), sol['A'], gamma, reward_scale).item():.4f}")

    xs = np.linspace(0.5, 5.0, 200)
    V_exact = analytical_value(xs, sol["A"], gamma, reward_scale)

    # ── Build problem ──────────────────────────────────────────────────
    # c_rate bounds: optimal k ≈ 0.048, range [0.005, 0.20] wide enough
    problem = MertonProblem(
        r_f=r_f, mu=mu, sigma=sigma, gamma=gamma, rho=rho,
        pi_bounds=(0.0, 1.5),
        crate_bounds=(0.005, 0.20),
        x_range=(0.5, 5.0),
        reward_scale=reward_scale,
    )

    # ── Policy iteration ──────────────────────────────────────────────
    cfg = PIConfig(
        rho=rho,
        T=20.0,
        dt=0.05,
        n_trajectories=256,
        n_eval_steps=400,
        n_improve_steps=40,
        lr_value=3e-3,
        lr_policy=3e-4,
        hidden_dim=128,
        n_layers=3,
        n_collocation=512,
        n_outer=40,
        device=device,
    )
    solver = PolicyIteration(problem, cfg)

    print("\n── Running policy iteration ──")
    history = solver.solve(verbose=True)

    # ── Compare learned vs analytical ─────────────────────────────────
    xs_t = torch.from_numpy(xs).float().unsqueeze(-1).to(device)
    with torch.no_grad():
        V_learned = solver.V_net(xs_t).cpu().numpy().flatten()
        a_learned = solver.policy_net(xs_t).cpu().numpy()   # (N, 2)

    pi_learned = a_learned[:, 0]
    cr_learned = a_learned[:, 1]                 # c_rate

    # ── Monte-Carlo validation ────────────────────────────────────────
    x0 = torch.tensor([1.0])
    V_pred, V_mc = solver.mc_validate(x0, n_sims=2048, T=60.0)
    V_true = analytical_value(np.array([1.0]), sol["A"], gamma, reward_scale).item()
    print(f"\n── Monte-Carlo validation at x₀ = 1.0 ──")
    print(f"  V_learned(x₀) = {V_pred:.4f}")
    print(f"  V_MC(x₀)      = {V_mc:.4f}")
    print(f"  V_exact(x₀)   = {V_true:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────
    plot_convergence(history)

    plot_value_comparison_1d(
        xs, V_exact, V_learned,
        title="Merton — Value Function (scaled)",
        xlabel="Wealth x",
    )

    # Policy: two panels (π, c_rate)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(xs, pi_learned, "b-", lw=2, label="Learned π(x)")
    axes[0].axhline(sol["pi_star"], color="r", ls="--", lw=2,
                     label=f"π* = {sol['pi_star']:.3f}")
    axes[0].set_xlabel("Wealth x")
    axes[0].set_ylabel("Risky fraction π")
    axes[0].set_title("Merton — Risky Fraction")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, cr_learned, "b-", lw=2, label="Learned c_rate(x)")
    axes[1].axhline(sol["k"], color="r", ls="--", lw=2,
                     label=f"k* = {sol['k']:.4f}")
    axes[1].set_xlabel("Wealth x")
    axes[1].set_ylabel("Consumption rate c/x")
    axes[1].set_title("Merton — Consumption Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Log-log plot of |V(x)| to verify slope = 1−γ
    fig, ax = plt.subplots(figsize=(6, 4))
    mask = xs > 0
    ax.plot(np.log(xs[mask]), np.log(-V_exact[mask]), "r--", lw=2, label="Exact (log)")
    V_safe = np.where(V_learned < 0, -V_learned, np.nan)
    valid = mask & np.isfinite(V_safe)
    ax.plot(np.log(xs[valid]), np.log(V_safe[valid]), "b-", lw=2, label="Learned (log)")
    ax.set_xlabel("log(x)")
    ax.set_ylabel("log(−V(x))")
    ax.set_title(f"Log-log slope should be {1 - gamma:.1f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
