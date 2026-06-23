"""
Neural PDE fitted to data with pure PyTorch:  Euler in time, P1 FEM in space.

We consider a 1-D semilinear reaction-diffusion equation on Omega = (0, 1) with
homogeneous Neumann (conormal) boundary conditions, in the sign convention of
Troeltzsch (2010), with the *right-hand side* parametrized by a neural network
(the analogue of the distributed control v in eq. (5.8) of the book, and of the
vector field f_theta in the Neural ODE):

        y_t + A y + d(x,t,y) = f_theta(x,t,y)   in  Q = Omega x (0, T)
        d_nu y               = 0                on  Sigma   (g = 0, b = 0)
        y(.,0)               = y_0              in  Omega

with the elliptic operator  A y = -nu * y_xx  (nu > 0) and known reaction d.  In
this example the known reaction is d == 0, so f_theta learns the *entire* source
term and should recover the ground-truth reaction directly.  Equivalently
        y_t = nu y_xx + f_theta(y).

Space discretization: piecewise-linear (P1) finite elements on a uniform mesh,
giving the mass matrix M and stiffness matrix K.  With mass lumping (diagonal
M_L) the semidiscrete system is

        M_L Y'(t) + nu K Y(t) = M_L f_theta(Y(t)),

i.e.  Y'(t) = -L Y(t) + f_theta(Y(t)) ,   L = nu M_L^{-1} K  (discrete A).
We march it with explicit Euler in time.  Backprop through the unrolled scheme
is the discrete adjoint of the equations derived in the post.

Loss (data fitting + Tikhonov / coercivity regularizer):

        J(theta) = (1/2) sum_{m,j,k} |y_theta(x_j,t_k; y_0^m) - y_obs|^2
                   + (gamma/2) |theta|^2 .

DATA COVERAGE.  A reaction f_theta(y) can only be identified on the range of
states the data visits.  We therefore simulate SEVERAL initial conditions whose
trajectories together sweep y in [0, ~1.2], which lets f_theta recover the
nonlinearity over the whole interval.

Ground-truth reaction: Fisher-KPP,  r(y) = 5 y (1 - y).
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
DTYPE = torch.float64          # FEM likes double precision


# -----------------------------------------------------------------------------
# 1. P1 finite-element assembly on a uniform mesh of (0, L)
# -----------------------------------------------------------------------------
def assemble_fem(n_nodes, L=1.0):
    h = L / (n_nodes - 1)
    x = torch.linspace(0.0, L, n_nodes, dtype=DTYPE)

    # Consistent mass matrix (tridiagonal): h/6 * [1 4 1]
    M = torch.zeros(n_nodes, n_nodes, dtype=DTYPE)
    K = torch.zeros(n_nodes, n_nodes, dtype=DTYPE)
    for e in range(n_nodes - 1):
        M[e, e]     += 2 * h / 6;  M[e, e + 1]     += h / 6
        M[e + 1, e] += h / 6;      M[e + 1, e + 1] += 2 * h / 6
        K[e, e]     += 1 / h;      K[e, e + 1]     += -1 / h
        K[e + 1, e] += -1 / h;     K[e + 1, e + 1] += 1 / h
    # Neumann BC => no constraint on K; natural conormal derivative = 0.
    Mlump = M.sum(dim=1)                       # row-sum lumping -> diagonal mass
    return x, M, K, Mlump


# Explicit Euler diffusion stability needs dt < h^2 / (2 nu).
# Here h = 1/50 = 0.02, nu = 0.05 -> dt < 0.004; we use dt = 1/600 ~ 0.0017.
N_NODES = 51
L = 1.0
NU = 0.05
x_nodes, M, K, Mlump = assemble_fem(N_NODES, L)
Linv = (1.0 / Mlump)[:, None] * K              # = M_L^{-1} K  (dense, fine in 1-D)
A_DISC = NU * Linv                             # discrete elliptic operator A


def fem_euler(source, Y0, t):
    """Solve M_L Y' + nu K Y = M_L f(Y) with explicit Euler. source: Y -> f(Y).
    Y0 may be batched: shape (B, N_nodes). Returns (N_t+1, B, N_nodes)."""
    Ys = [Y0]
    Y = Y0
    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        Y = Y + dt * (-(Y @ A_DISC.t()) + source(Y))
        Ys.append(Y)
    return torch.stack(Ys, dim=0)


# -----------------------------------------------------------------------------
# 2. Ground-truth data: several initial conditions for broad state coverage
# -----------------------------------------------------------------------------
def f_true(Y):
    # Fisher-KPP reaction r(y) = 5 y (1-y)  (the source term we want to recover)
    return 5.0 * Y * (1.0 - Y)


T = 1.0
N_T = 600
t_grid = torch.linspace(0.0, T, N_T + 1, dtype=DTYPE)
xc = x_nodes

# initial conditions chosen so the trajectories together visit y in [0, ~1.2]
ic_list = [
    0.5 * (1 + torch.cos(2 * np.pi * (xc - 0.5))) * (torch.abs(xc - 0.5) < 0.5),
    0.20 + 0.10 * torch.cos(2 * np.pi * xc),
    0.05 + 0.05 * torch.sin(np.pi * xc),
    0.40 * torch.sin(np.pi * xc),
    0.90 - 0.30 * torch.cos(2 * np.pi * xc),
    1.10 + 0.10 * torch.cos(2 * np.pi * xc),
]
Y0 = torch.stack(ic_list, dim=0).to(DTYPE)     # (n_ic, N_nodes)

with torch.no_grad():
    Y_clean = fem_euler(f_true, Y0, t_grid)            # (N_t+1, n_ic, N_nodes)
    Y_obs = Y_clean + 0.005 * torch.randn_like(Y_clean)


# -----------------------------------------------------------------------------
# 3. Neural source f_theta(y)  (acts pointwise on nodal values)
# -----------------------------------------------------------------------------
class Source(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, Y):                       # Y: (..., N_nodes)
        return self.net(Y.unsqueeze(-1)).squeeze(-1)


model = Source().to(dtype=DTYPE)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.3)
GAMMA = 1e-6
EPOCHS = 4000

obs_idx = torch.arange(0, N_T + 1, 8)          # subsample in time -> data points
history = []
for epoch in range(EPOCHS):
    opt.zero_grad()
    Y_pred = fem_euler(model, Y0, t_grid)
    data_loss = 0.5 * ((Y_pred[obs_idx] - Y_obs[obs_idx]) ** 2).sum(dim=-1).mean()
    reg = 0.5 * GAMMA * sum((p ** 2).sum() for p in model.parameters())
    loss = data_loss + reg
    loss.backward()
    opt.step()
    sched.step()
    history.append(data_loss.item())
    if epoch % 200 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:4d}   data_loss {data_loss.item():.5e}")

with torch.no_grad():
    Y_fit = fem_euler(model, Y0, t_grid)


# -----------------------------------------------------------------------------
# 4. Plots
# -----------------------------------------------------------------------------
xn, tn = x_nodes.numpy(), t_grid.numpy()
ic_show = 0                                     # which IC to display as a field
Yo, Yf = Y_obs[:, ic_show].numpy(), Y_fit[:, ic_show].numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

im0 = axes[0].pcolormesh(tn, xn, Yo.T, shading="auto", cmap="viridis")
axes[0].set_title("data  $y_{obs}(x,t)$"); axes[0].set_xlabel("$t$"); axes[0].set_ylabel("$x$")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(tn, xn, Yf.T, shading="auto", cmap="viridis")
axes[1].set_title(r"fit  $y_\theta(x,t)$"); axes[1].set_xlabel("$t$"); axes[1].set_ylabel("$x$")
fig.colorbar(im1, ax=axes[1])

# recovered source vs truth, shaded over the range of states the data visits
yvis_lo = float(Y_obs.min()); yvis_hi = float(Y_obs.max())
yy = torch.linspace(-0.1, 1.3, 300, dtype=DTYPE).unsqueeze(-1)
with torch.no_grad():
    f_learned = model.net(yy).squeeze(-1).numpy()
yy = yy.squeeze(-1).numpy()
axes[2].axvspan(yvis_lo, yvis_hi, color="0.9", label="data range")
axes[2].plot(yy, f_true(torch.tensor(yy)).numpy(), "k--", lw=2, label="true $r(y)$")
axes[2].plot(yy, f_learned, "C3", lw=2, label=r"learned $f_\theta(y)$")
axes[2].set_title("recovered source"); axes[2].set_xlabel("$y$"); axes[2].set_ylabel("$f(y)$")
axes[2].legend()

fig.tight_layout()
out = "../../content/posts/neural-pde/neural_pde_fit.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)

fig2, ax = plt.subplots(figsize=(5.5, 3.8))
ax.semilogy(history, color="C2")
ax.set_xlabel("epoch"); ax.set_ylabel("data loss"); ax.set_title("Neural PDE training")
fig2.tight_layout()
out2 = "../../content/posts/neural-pde/neural_pde_loss.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print("saved", out2)
