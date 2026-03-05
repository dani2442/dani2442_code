"""Howard / Policy Iteration for continuous-time stochastic control.

Algorithm
---------
For k = 0, 1, 2, …:
  1) **Policy evaluation**  – estimate  V_k = V^{α_k}  via Monte-Carlo
     (Feynman-Kac: simulate closed-loop SDE, compute discounted returns,
      regress V_θ against those returns).
  2) **Q-function** – form
         Q_k(x,a) = (1/ρ)[r(x,a) + L^a V_k(x)]
     using the *known* Σ to get the full generator
         L^a V = ∇V · f(x,a) + ½ Tr(ΣΣᵀ(x,a) ∇²V).
  3) **Policy improvement** – update α_{k+1} by gradient ascent on Q_k:
         α_{k+1}(x) ∈ argmax_a Q_k(x,a).
  4) **Stopping** – check HJB residual → 0.

The diffusion coefficient Σ(x,a) is assumed *known* (given by the problem).
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torchsde

from model import ValueNetwork, PolicyNetwork


# ---------------------------------------------------------------------------
# Abstract control problem
# ---------------------------------------------------------------------------

class ControlProblem(abc.ABC):
    """Interface that any continuous-time control problem must implement."""

    @property
    @abc.abstractmethod
    def state_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def action_dim(self) -> int: ...

    @abc.abstractmethod
    def drift(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """f(x,a) → (batch, d)."""

    @abc.abstractmethod
    def diffusion(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Σ(x,a) → (batch, d, m)  [full matrix form for the generator]."""

    @abc.abstractmethod
    def reward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """r(x,a) → (batch, 1)."""

    @abc.abstractmethod
    def sample_initial_states(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample x_0 ∈ ℝ^d → (n, d)."""

    @abc.abstractmethod
    def action_low(self, device: torch.device) -> torch.Tensor:
        """Lower bound of compact action set → (m,)."""

    @abc.abstractmethod
    def action_high(self, device: torch.device) -> torch.Tensor:
        """Upper bound → (m,)."""

    @property
    def noise_type(self) -> str:
        """torchsde noise type: 'diagonal' | 'additive' | 'general'."""
        return "diagonal"

    @property
    def state_clamp(self) -> float | None:
        """Max absolute state value.  None = no clamping."""
        return None

    # ----- helpers for torchsde g() shape -----------------------------------

    def diffusion_sde(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return the diffusion in the shape that torchsde expects.

        Override if the default (derived from self.diffusion) is wrong.
        Default: diagonal noise → (batch, d).
        """
        # For diagonal noise Σ is (batch, d, d) diagonal → just take diag
        Sigma = self.diffusion(x, a)  # (batch, d, m)
        if self.noise_type == "diagonal":
            # Assume d == m and Σ is diagonal → extract diagonal
            return torch.diagonal(Sigma, dim1=-2, dim2=-1)  # (batch, d)
        return Sigma  # general / additive: (batch, d, m)


# ---------------------------------------------------------------------------
# Closed-loop SDE wrapper for torchsde
# ---------------------------------------------------------------------------

class ClosedLoopSDE(torchsde.SDEIto):
    """SDE  dX = f(X, α(X)) dt + Σ(X, α(X)) dW  driven by a policy."""

    def __init__(self, problem: ControlProblem, policy_fn, noise_type: str = "diagonal"):
        super().__init__(noise_type=noise_type)
        self.problem = problem
        self.policy_fn = policy_fn
        self._clamp = problem.state_clamp
        self._clamp_lo = getattr(problem, "state_clamp_lo", None)

    # torchsde interface
    sde_type = "ito"

    def _clamp_state(self, y):
        if self._clamp is not None:
            lo = self._clamp_lo if self._clamp_lo is not None else -self._clamp
            y = y.clamp(lo, self._clamp)
        return y

    def f(self, t, y):
        y = self._clamp_state(y)
        a = self.policy_fn(y)
        return self.problem.drift(y, a)

    def g(self, t, y):
        y = self._clamp_state(y)
        a = self.policy_fn(y)
        return self.problem.diffusion_sde(y, a)


# ---------------------------------------------------------------------------
# Generator  L^a V  (uses known Σ)
# ---------------------------------------------------------------------------

def compute_generator(
    V_net: ValueNetwork,
    x: torch.Tensor,
    f_xa: torch.Tensor,
    Sigma_xa: torch.Tensor,
) -> torch.Tensor:
    """Compute  L^a V(x) = ∇V·f + ½ Tr(ΣΣᵀ ∇²V).

    Parameters
    ----------
    V_net   : value network  V_θ
    x       : (batch, d)  **must** have requires_grad
    f_xa    : (batch, d)  drift   f(x, a)  — may carry grad w.r.t. action
    Sigma_xa: (batch, d, m) diffusion Σ(x, a)

    Returns
    -------
    L_V : (batch, 1)
    """
    V = V_net(x)  # (batch, 1)
    # ∇V  (batch, d)
    grad_V = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

    # Drift term  ∇V · f
    drift_term = (grad_V * f_xa).sum(dim=-1, keepdim=True)  # (batch, 1)

    # Hessian-diffusion term  ½ Tr(A H)  with A = ΣΣᵀ
    d = x.shape[1]
    # Build Hessian rows: H_{ij} = ∂²V / ∂x_i ∂x_j
    hessian_rows = []
    for i in range(d):
        row = torch.autograd.grad(grad_V[:, i].sum(), x, create_graph=True)[0]
        hessian_rows.append(row)  # each (batch, d)
    H = torch.stack(hessian_rows, dim=1)  # (batch, d, d)

    # A = Σ Σᵀ  (batch, d, d)
    A = torch.bmm(Sigma_xa, Sigma_xa.transpose(1, 2))

    # ½ Tr(A H)  = ½ Σ_{ij} A_{ij} H_{ij}
    diff_term = 0.5 * (A * H).sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)

    return drift_term + diff_term  # (batch, 1)


def compute_generator_detached(
    V_net: ValueNetwork,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute ∇V and ∇²V at *x*, detaching from V_net parameters.

    Used for policy improvement so gradients flow only through the
    action-dependent terms (f, Σ, r).

    Returns
    -------
    grad_V : (batch, d)  detached
    H      : (batch, d, d)  detached
    """
    x = x.detach().requires_grad_(True)
    V = V_net(x)  # (batch, 1)
    grad_V = torch.autograd.grad(V.sum(), x, create_graph=True)[0]  # (batch, d)

    d = x.shape[1]
    hessian_rows = []
    for i in range(d):
        row = torch.autograd.grad(grad_V[:, i].sum(), x, retain_graph=True)[0]
        hessian_rows.append(row)
    H = torch.stack(hessian_rows, dim=1)  # (batch, d, d)

    return grad_V.detach(), H.detach()


def generator_from_precomputed(
    grad_V: torch.Tensor,
    H: torch.Tensor,
    f_xa: torch.Tensor,
    Sigma_xa: torch.Tensor,
) -> torch.Tensor:
    """L^a V using pre-computed (detached) ∇V and H.

    Gradients flow through f_xa and Sigma_xa only (i.e. through the policy).
    """
    drift_term = (grad_V * f_xa).sum(dim=-1, keepdim=True)
    A = torch.bmm(Sigma_xa, Sigma_xa.transpose(1, 2))
    diff_term = 0.5 * (A * H).sum(dim=(-2, -1)).unsqueeze(-1)
    return drift_term + diff_term


# ---------------------------------------------------------------------------
# Policy Iteration solver
# ---------------------------------------------------------------------------

@dataclass
class PIConfig:
    """Hyper-parameters for policy iteration."""
    rho: float = 0.1              # discount rate
    T: float = 8.0                # episode horizon for MC rollouts
    dt: float = 0.05              # SDE integration step
    n_trajectories: int = 128     # parallel trajectories per evaluation
    n_eval_steps: int = 200       # SGD steps for value fitting
    n_improve_steps: int = 30     # SGD steps for policy improvement (keep small!)
    lr_value: float = 3e-3
    lr_policy: float = 3e-4       # conservative to avoid overshooting
    hidden_dim: int = 64
    n_layers: int = 2
    n_collocation: int = 512      # points sampled for policy improvement
    n_outer: int = 25             # outer policy-iteration loops
    tau: float = 0.0              # soft-update coefficient (0 = hard copy)
    warmup_iters: int = 3         # ramp improve steps from 1 → n_improve_steps
    device: str = "cpu"


class PolicyIteration:
    """Howard policy iteration for continuous-time stochastic control."""

    def __init__(self, problem: ControlProblem, cfg: PIConfig):
        self.problem = problem
        self.cfg = cfg
        dev = torch.device(cfg.device)
        self.device = dev

        # Networks
        self.V_net = ValueNetwork(
            problem.state_dim, cfg.hidden_dim, cfg.n_layers
        ).to(dev)
        self.policy_net = PolicyNetwork(
            problem.state_dim,
            problem.action_dim,
            problem.action_low(dev),
            problem.action_high(dev),
            cfg.hidden_dim,
            cfg.n_layers,
        ).to(dev)

        # Optimizers
        self.opt_V = torch.optim.Adam(self.V_net.parameters(), lr=cfg.lr_value)
        self.opt_pi = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr_policy)

        # Store trajectory states for policy improvement collocation
        self._traj_states: torch.Tensor | None = None

        # Logging
        self.history: dict[str, list[float]] = {
            "eval_loss": [],
            "improve_loss": [],
            "hjb_residual": [],
        }

    # ------------------------------------------------------------------
    # 1) Policy evaluation  (Monte-Carlo Feynman-Kac)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _simulate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate closed-loop trajectories and compute discounted returns.

        Uses terminal-value bootstrapping:
            V̂(X_t) ≈ Σ_{j=t}^{T/dt} e^{-ρ(j-t)dt} r_j dt + e^{-ρ(T-t)} V_θ(X_T)

        Returns
        -------
        states  : (N, d)   all intermediate states used as regression inputs
        returns : (N, 1)   corresponding discounted-return targets
        """
        cfg = self.cfg
        dev = self.device

        ts = torch.linspace(0, cfg.T, int(cfg.T / cfg.dt) + 1, device=dev)
        n_steps = len(ts) - 1

        x0 = self.problem.sample_initial_states(cfg.n_trajectories, dev)  # (B, d)

        # Build closed-loop SDE
        policy_fn = lambda y: self.policy_net(y)
        sde = ClosedLoopSDE(self.problem, policy_fn, self.problem.noise_type)

        traj = torchsde.sdeint(sde, x0, ts, method="euler", dt=cfg.dt)
        # traj: (n_steps+1, B, d)

        # Compute rewards at each step — vectorised
        all_states = traj[:-1]   # (n_steps, B, d)
        S, B, d = all_states.shape
        flat_s = all_states.reshape(S * B, d)
        flat_a = self.policy_net(flat_s)           # (S*B, m)
        flat_r = self.problem.reward(flat_s, flat_a)  # (S*B, 1)
        all_rewards = flat_r.reshape(S, B, 1)      # (n_steps, B, 1)

        # Terminal value bootstrap: V_θ(X_T)
        terminal_states = traj[-1]  # (B, d)
        V_terminal = self.V_net(terminal_states)  # (B, 1)

        # Discounted rewards: disc_r[t] = e^{-ρ t dt} r(X_t) dt
        steps = torch.arange(n_steps, device=dev, dtype=torch.float32)
        disc_factor = torch.exp(-cfg.rho * steps * cfg.dt)  # (n_steps,)
        disc_r = disc_factor.unsqueeze(-1).unsqueeze(-1) * all_rewards * cfg.dt

        # Add discounted terminal value as a "final reward"
        disc_terminal = math.exp(-cfg.rho * cfg.T) * V_terminal  # (B, 1)

        # V̂(X_t) = (1/e^{-ρt}) [Σ_{j≥t} disc_r[j] + disc_terminal]
        cum_from_end = disc_r.flip(0).cumsum(0).flip(0)  # (n_steps, B, 1)
        returns = (cum_from_end + disc_terminal.unsqueeze(0)) / disc_factor.unsqueeze(-1).unsqueeze(-1)

        # Flatten
        states_flat = all_states.reshape(S * B, d)
        returns_flat = returns.reshape(S * B, 1)

        return states_flat, returns_flat

    def evaluate_policy(self) -> float:
        """Step 1: fit V_θ to MC discounted returns under current policy."""
        states, returns = self._simulate()
        self._traj_states = states.detach()  # cache for policy improvement
        n = states.shape[0]
        batch_size = min(2048, n)

        self.V_net.train()
        total_loss = 0.0
        n_batches = 0
        for _ in range(self.cfg.n_eval_steps):
            idx = torch.randint(n, (batch_size,), device=self.device)
            xb, yb = states[idx], returns[idx]
            pred = self.V_net(xb)
            loss = nn.functional.mse_loss(pred, yb)
            self.opt_V.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.V_net.parameters(), 1.0)
            self.opt_V.step()
            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        self.history["eval_loss"].append(avg)
        return avg

    # ------------------------------------------------------------------
    # 2-3) Q-function + policy improvement
    # ------------------------------------------------------------------

    def improve_policy(self, n_steps: int | None = None) -> float:
        """Steps 2-3: improve α_φ by maximising Q_k(x, α_φ(x)).

        Q_k(x,a) = (1/ρ)[r(x,a) + L^a V_k(x)]
        with L^a V computed from known Σ and pre-computed (detached) ∇V, H.
        Gradients flow only through the policy network.
        """
        cfg = self.cfg
        dev = self.device
        n_steps = n_steps or cfg.n_improve_steps

        self.V_net.eval()
        self.policy_net.train()

        total_loss = 0.0
        for _ in range(n_steps):
            # Sample collocation from trajectory distribution (+ some fresh)
            if self._traj_states is not None and len(self._traj_states) >= cfg.n_collocation:
                n_traj = int(0.8 * cfg.n_collocation)
                n_fresh = cfg.n_collocation - n_traj
                idx = torch.randint(len(self._traj_states), (n_traj,), device=dev)
                x_traj = self._traj_states[idx]
                x_fresh = self.problem.sample_initial_states(n_fresh, dev)
                x = torch.cat([x_traj, x_fresh], dim=0)
            else:
                x = self.problem.sample_initial_states(cfg.n_collocation, dev)

            # Pre-compute ∇V, H (detached from V_net)
            grad_V, H = compute_generator_detached(self.V_net, x)

            # Forward policy → action (carries grad w.r.t. φ)
            a = self.policy_net(x)

            # Compute Q_k(x, a)
            r_xa = self.problem.reward(x, a)         # (batch, 1)
            f_xa = self.problem.drift(x, a)           # (batch, d)
            Sigma_xa = self.problem.diffusion(x, a)   # (batch, d, m)

            L_V = generator_from_precomputed(grad_V, H, f_xa, Sigma_xa)
            Q = (r_xa + L_V) / cfg.rho  # (batch, 1)

            loss = -Q.mean()  # maximise Q
            self.opt_pi.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.opt_pi.step()
            total_loss += loss.item()

        avg = total_loss / max(n_steps, 1)
        self.history["improve_loss"].append(avg)
        return avg

    # ------------------------------------------------------------------
    # 4) HJB residual: ρV(x) − max_a [r + L^a V]
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_hjb_residual(self, n_test: int = 512, n_action_samples: int = 64) -> float:
        """Estimate  sup_x |ρ V(x) − max_a[r + L^a V]|  by sampling."""
        dev = self.device
        x = self.problem.sample_initial_states(n_test, dev)

        # We need grad for Hessian, so temporarily enable grad
        with torch.enable_grad():
            grad_V, H = compute_generator_detached(self.V_net, x)

        rho_V = self.cfg.rho * self.V_net(x)  # (n_test, 1)

        # Sample actions and find best  r + L^a V
        a_lo = self.problem.action_low(dev)
        a_hi = self.problem.action_high(dev)
        m = self.problem.action_dim

        # Also include the current policy action
        a_policy = self.policy_net(x)  # (n_test, m)
        best_rhs = torch.full((n_test, 1), -float("inf"), device=dev)

        # Batched random actions
        a_rand = a_lo + (a_hi - a_lo) * torch.rand(n_test, n_action_samples, m, device=dev)
        # Prepend the policy action
        all_a = torch.cat([a_policy.unsqueeze(1), a_rand], dim=1)  # (n_test, 1+S, m)

        for j in range(all_a.shape[1]):
            a_j = all_a[:, j]  # (n_test, m)
            r_j = self.problem.reward(x, a_j)
            f_j = self.problem.drift(x, a_j)
            S_j = self.problem.diffusion(x, a_j)
            L_j = generator_from_precomputed(grad_V, H, f_j, S_j)
            rhs_j = r_j + L_j
            best_rhs = torch.max(best_rhs, rhs_j)

        residual = (rho_V - best_rhs).abs().mean().item()
        self.history["hjb_residual"].append(residual)
        return residual

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = True):
        """Run the full policy-iteration loop."""
        cfg = self.cfg
        for k in range(cfg.n_outer):
            eval_loss = self.evaluate_policy()
            # Warmup: linearly ramp improve steps over the first few iters
            if cfg.warmup_iters > 0 and k < cfg.warmup_iters:
                frac = (k + 1) / cfg.warmup_iters
                n_imp = max(1, int(frac * cfg.n_improve_steps))
            else:
                n_imp = cfg.n_improve_steps
            imp_loss = self.improve_policy(n_steps=n_imp)
            res = self.compute_hjb_residual()
            if verbose:
                print(
                    f"PI iter {k+1:3d}/{cfg.n_outer} │ "
                    f"eval loss {eval_loss:.4e} │ "
                    f"improve obj {imp_loss:+.4e} │ "
                    f"HJB residual {res:.4e}"
                )
        return self.history

    # ------------------------------------------------------------------
    # Monte-Carlo validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def mc_validate(
        self,
        x0: torch.Tensor,
        n_sims: int = 1024,
        T: float | None = None,
    ) -> tuple[float, float]:
        """Compare V_θ(x0) to mean discounted MC return under learned policy.

        Parameters
        ----------
        x0    : (d,)  single initial state
        n_sims: number of independent simulations

        Returns
        -------
        (V_predicted, V_mc)
        """
        dev = self.device
        cfg = self.cfg
        T_eval = T or cfg.T * 3
        ts = torch.linspace(0, T_eval, int(T_eval / cfg.dt) + 1, device=dev)
        n_steps = len(ts) - 1

        x0_batch = x0.unsqueeze(0).expand(n_sims, -1).to(dev)

        sde = ClosedLoopSDE(self.problem, self.policy_net, self.problem.noise_type)
        traj = torchsde.sdeint(sde, x0_batch, ts, method="euler", dt=cfg.dt)
        # (n_steps+1, n_sims, d)

        # Vectorised rewards
        all_states = traj[:-1]  # (n_steps, n_sims, d)
        S, B, d = all_states.shape
        flat_s = all_states.reshape(S * B, d)
        flat_a = self.policy_net(flat_s)
        flat_r = self.problem.reward(flat_s, flat_a)
        rewards = flat_r.reshape(S, B, 1)

        disc = torch.exp(-cfg.rho * torch.arange(n_steps, device=dev).float() * cfg.dt)
        mc_returns = (disc.unsqueeze(-1).unsqueeze(-1) * rewards).sum(dim=0) * cfg.dt
        V_mc = mc_returns.mean().item()

        V_pred = self.V_net(x0.unsqueeze(0).to(dev)).item()
        return V_pred, V_mc
