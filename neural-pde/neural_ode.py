"""
Neural ODE fitted to data with pure PyTorch.

We model the dynamics of a state y(t) in R^n by a neural network f_theta and
integrate the IVP

        y'(t) = f_theta(y(t), t),   t in (0, T],     y(0) = y_0

with the explicit (forward) Euler scheme.  Parameters theta are fitted to
observed trajectory data by minimizing the regularized tracking loss

        J(theta) = (1/2) sum_k |y_theta(t_k) - y_obs_k|^2  +  (gamma/2) |theta|^2 .

The Tikhonov term (gamma/2)|theta|^2 makes the reduced functional coercive in
theta, guaranteeing a minimizer (see the appendix of the post).

Backpropagation through the unrolled Euler steps is exactly the *discrete
adjoint* method, so we let autograd compute nabla_theta J for us.

Ground truth: the nonlinear spiral of Chen et al. (2018),
        y' = (y^3) A^T,   A = [[-0.1, 2.0], [-2.0, -0.1]].
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cpu"
DTYPE = torch.float32

# -----------------------------------------------------------------------------
# 1. Ground-truth dynamics and data generation
# -----------------------------------------------------------------------------
A_TRUE = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], dtype=DTYPE)


def f_true(y):
    # y: (..., 2);   y' = (y^3) A^T
    return torch.matmul(y ** 3, A_TRUE.t())


def euler_rollout(f, y0, t):
    """Integrate y' = f(y, t) with explicit Euler on the grid t. Returns (N+1, dim)."""
    ys = [y0]
    y = y0
    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        y = y + dt * f(y, t[k])
        ys.append(y)
    return torch.stack(ys, dim=0)


# Explicit Euler on the stiff cubic dynamics needs a small step: dt = 6/1200
# = 0.005 is stable, dt = 0.02 blows up.
T = 6.0
N = 1200                      # integration / observation steps
t_grid = torch.linspace(0.0, T, N + 1, dtype=DTYPE)
y0 = torch.tensor([2.0, 0.0], dtype=DTYPE)

with torch.no_grad():
    y_clean = euler_rollout(lambda y, t: f_true(y), y0, t_grid)
    noise = 0.02 * torch.randn_like(y_clean)
    y_obs = y_clean + noise          # noisy observations = the "data" D


# -----------------------------------------------------------------------------
# 2. Neural vector field f_theta
# -----------------------------------------------------------------------------
class ODEFunc(nn.Module):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        # smooth (C^infty) Lipschitz activations -> f_theta is locally Lipschitz,
        # which is what Picard-Lindelof needs for well-posedness (see appendix).
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        # start with f_theta ~ 0 so the initial rollout cannot blow up
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y, t):
        return self.net(y)


# -----------------------------------------------------------------------------
# 3. Training: minimize J(theta)
# -----------------------------------------------------------------------------
func = ODEFunc().to(DEVICE)
opt = torch.optim.Adam(func.parameters(), lr=7e-3)
GAMMA = 1e-4                  # Tikhonov / coercivity weight
ITERS = 1000
BATCH_TIME = 40               # length of each sub-trajectory
BATCH_SIZE = 32               # number of sub-trajectories per step


def get_batch():
    # standard Neural-ODE mini-batching (Chen et al. 2018): integrate many short
    # sub-trajectories starting from points along the observed data. This is far
    # easier to optimize than one long stiff rollout.
    s = torch.randint(0, N - BATCH_TIME, (BATCH_SIZE,))
    y_b0 = y_obs[s]                                            # (B, dim)
    t_b = t_grid[:BATCH_TIME]
    y_btarget = torch.stack([y_obs[s + i] for i in range(BATCH_TIME)], 0)  # (BT, B, dim)
    return y_b0, t_b, y_btarget


import copy

history = []
best_loss = float("inf")
best_state = copy.deepcopy(func.state_dict())
for it in range(ITERS):
    opt.zero_grad()
    y_b0, t_b, y_btarget = get_batch()
    y_bpred = euler_rollout(func, y_b0, t_b)                   # (BT, B, dim)
    data_loss = 0.5 * ((y_bpred - y_btarget) ** 2).sum(dim=-1).mean()
    reg = 0.5 * GAMMA * sum((p ** 2).sum() for p in func.parameters())
    loss = data_loss + reg
    loss.backward()
    torch.nn.utils.clip_grad_norm_(func.parameters(), 1.0)
    opt.step()
    if it % 25 == 0:
        # evaluate the full-trajectory fit and keep the best checkpoint
        with torch.no_grad():
            full = euler_rollout(func, y0, t_grid)
            full_loss = 0.5 * ((full - y_obs) ** 2).sum(dim=1).mean().item()
        history.append(full_loss)
        if full_loss < best_loss:
            best_loss = full_loss
            best_state = copy.deepcopy(func.state_dict())
        if it % 200 == 0 or it == ITERS - 1:
            print(f"iter {it:4d}   batch_loss {data_loss.item():.4e}   traj_loss {full_loss:.4e}")

func.load_state_dict(best_state)                               # use best model for the figures
print(f"best trajectory loss: {best_loss:.4e}")
with torch.no_grad():
    y_fit = euler_rollout(func, y0, t_grid)


# -----------------------------------------------------------------------------
# 4. Plots
# -----------------------------------------------------------------------------
y_obs_np, y_fit_np, t_np = y_obs.numpy(), y_fit.numpy(), t_grid.numpy()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(y_obs_np[:, 0], y_obs_np[:, 1], ".", ms=3, alpha=0.4, label="data")
ax.plot(y_fit_np[:, 0], y_fit_np[:, 1], "-", lw=2, color="C3", label=r"$y_\theta$ (fit)")
ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$")
ax.set_title("Phase portrait"); ax.legend(); ax.set_aspect("equal", "box")

ax = axes[1]
ax.plot(t_np, y_obs_np[:, 0], ".", ms=3, alpha=0.4, color="C0")
ax.plot(t_np, y_obs_np[:, 1], ".", ms=3, alpha=0.4, color="C1")
ax.plot(t_np, y_fit_np[:, 0], "-", lw=2, color="C0", label=r"$y_{\theta,1}$")
ax.plot(t_np, y_fit_np[:, 1], "-", lw=2, color="C1", label=r"$y_{\theta,2}$")
ax.set_xlabel("$t$"); ax.set_title("Time series"); ax.legend()

fig.tight_layout()
out = "../../content/posts/neural-pde/neural_ode_fit.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)

fig2, ax = plt.subplots(figsize=(5.5, 3.8))
ax.semilogy(np.arange(len(history)) * 25, history, color="C2")
ax.set_xlabel("iteration"); ax.set_ylabel("trajectory loss"); ax.set_title("Neural ODE training")
fig2.tight_layout()
out2 = "../../content/posts/neural-pde/neural_ode_loss.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print("saved", out2)
