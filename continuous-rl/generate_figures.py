"""Generate all blog-post figures for both examples.

Saves PNGs to the post folder next to index.md.
"""
from __future__ import annotations
import sys, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from policy_iteration import PolicyIteration, PIConfig, ClosedLoopSDE
from example_lqr import StochasticLQR, preset_1d, solve_are, analytical_value, analytical_policy
from example_merton import MertonProblem, analytical_merton, analytical_value as merton_analytical_value
import torchsde

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "content", "posts", "continuous-rl")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
})

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# =========================================================================
# EXAMPLE 1: Stochastic LQR
# =========================================================================
print("\n" + "="*60)
print("EXAMPLE 1: Stochastic LQR")
print("="*60)

rho_lqr = 0.1
problem_lqr, params_lqr = preset_1d(sigma=0.3)
P, c, K = solve_are(**params_lqr, rho=rho_lqr)

cfg_lqr = PIConfig(
    rho=rho_lqr, T=8.0, dt=0.05,
    n_trajectories=128, n_eval_steps=200, n_improve_steps=30,
    lr_value=3e-3, lr_policy=3e-4,
    hidden_dim=64, n_layers=2,
    n_collocation=512, n_outer=25, device=device,
)
solver_lqr = PolicyIteration(problem_lqr, cfg_lqr)
history_lqr = solver_lqr.solve(verbose=True)

xs_lqr = np.linspace(-2, 2, 200).reshape(-1, 1)
V_exact_lqr = analytical_value(xs_lqr, P, c).flatten()
a_exact_lqr = analytical_policy(xs_lqr, K).flatten()

xs_t = torch.from_numpy(xs_lqr).float().to(device)
with torch.no_grad():
    V_learned_lqr = solver_lqr.V_net(xs_t).cpu().numpy().flatten()
    a_learned_lqr = solver_lqr.policy_net(xs_t).cpu().numpy().flatten()

# --- LQR Fig 1: Value + Policy comparison ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(xs_lqr.flatten(), V_exact_lqr, "r--", lw=2, label=r"$V^*(x)$ exact")
axes[0].plot(xs_lqr.flatten(), V_learned_lqr, "b-", lw=2, label=r"$V_\theta(x)$ learned")
axes[0].set_xlabel("State $x$"); axes[0].set_ylabel("$V(x)$")
axes[0].set_title("Value Function"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(xs_lqr.flatten(), a_exact_lqr, "r--", lw=2, label=r"$a^*(x)$ exact")
axes[1].plot(xs_lqr.flatten(), a_learned_lqr, "b-", lw=2, label=r"$\alpha_\phi(x)$ learned")
axes[1].set_xlabel("State $x$"); axes[1].set_ylabel("Action $a$")
axes[1].set_title("Optimal Policy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle("Stochastic LQR — Learned vs Exact", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lqr_value_policy.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved lqr_value_policy.png")

# --- LQR Fig 2: Convergence ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(history_lqr["eval_loss"], "o-", color="tab:blue", ms=3, lw=1.5)
axes[0].set_ylabel("Value-fit MSE"); axes[0].set_xlabel("PI iteration")
axes[0].set_yscale("log"); axes[0].set_title("Value-fit MSE"); axes[0].grid(True, alpha=0.3)

axes[1].plot(history_lqr["improve_loss"], "o-", color="tab:orange", ms=3, lw=1.5)
axes[1].set_ylabel(r"$-\mathbb{E}[Q]$"); axes[1].set_xlabel("PI iteration")
axes[1].set_title("Policy Objective"); axes[1].grid(True, alpha=0.3)

axes[2].plot(history_lqr["hjb_residual"], "o-", color="tab:red", ms=3, lw=1.5)
axes[2].set_ylabel("HJB residual"); axes[2].set_xlabel("PI iteration")
axes[2].set_yscale("log"); axes[2].set_title("HJB Residual"); axes[2].grid(True, alpha=0.3)
plt.suptitle("Stochastic LQR — Convergence", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lqr_convergence.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved lqr_convergence.png")

# --- LQR Fig 3: Sample trajectories + cumulative reward ---
n_samples = 5
with torch.no_grad():
    x0_lqr = torch.full((n_samples, 1), 1.5, device=device)
    T_sim = 10.0
    dt_sim = 0.05
    ts_sim = torch.linspace(0, T_sim, int(T_sim / dt_sim) + 1, device=device)
    sde_lqr = ClosedLoopSDE(problem_lqr, solver_lqr.policy_net, problem_lqr.noise_type)
    traj_lqr = torchsde.sdeint(sde_lqr, x0_lqr, ts_sim, method="euler", dt=dt_sim)
    # traj_lqr: (steps+1, n_samples, 1)
    states_np = traj_lqr[:, :, 0].cpu().numpy()  # (steps+1, n_samples)
    ts_np = ts_sim.cpu().numpy()

    # Compute instantaneous rewards along each trajectory
    st_flat = traj_lqr[:-1].reshape(-1, 1)  # (n_steps*n_samples, 1)
    act_flat = solver_lqr.policy_net(st_flat)
    rew_flat = problem_lqr.reward(st_flat, act_flat).cpu().numpy().flatten()
    n_steps_lqr = len(ts_np) - 1
    rew_all = rew_flat.reshape(n_steps_lqr, n_samples)  # (n_steps, n_samples)
    disc = np.exp(-rho_lqr * ts_np[:-1])
    cum_reward = np.cumsum(disc[:, None] * rew_all * dt_sim, axis=0)  # (n_steps, n_samples)

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
colors = plt.cm.tab10(np.linspace(0, 1, n_samples))
for i in range(n_samples):
    axes[0].plot(ts_np, states_np[:, i], lw=1.2, color=colors[i], alpha=0.7)
axes[0].axhline(0, color="gray", ls=":", lw=1)
axes[0].set_ylabel("State $x_t$"); axes[0].set_title(f"Optimal Trajectories ($x_0={1.5}$, {n_samples} samples)")
axes[0].grid(True, alpha=0.3)

for i in range(n_samples):
    axes[1].plot(ts_np[:-1], cum_reward[:, i], lw=1.2, color=colors[i], alpha=0.7)
axes[1].set_xlabel("Time $t$"); axes[1].set_ylabel("Cumulative discounted reward")
axes[1].set_title("Cumulative Discounted Reward"); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lqr_trajectory.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved lqr_trajectory.png")


# =========================================================================
# EXAMPLE 2: Merton Portfolio/Consumption
# =========================================================================
print("\n" + "="*60)
print("EXAMPLE 2: Merton Portfolio/Consumption")
print("="*60)

r_f, mu, sigma_m, gamma, rho_m = 0.03, 0.08, 0.20, 2.0, 0.05
reward_scale = 0.01
sol_m = analytical_merton(r_f, mu, sigma_m, gamma, rho_m)
print(f"  pi*={sol_m['pi_star']:.4f}, k*={sol_m['k']:.4f}")

problem_m = MertonProblem(
    r_f=r_f, mu=mu, sigma=sigma_m, gamma=gamma, rho=rho_m,
    pi_bounds=(0.0, 1.5), crate_bounds=(0.005, 0.20),
    x_range=(0.2, 10.0), reward_scale=reward_scale,
)

cfg_m = PIConfig(
    rho=rho_m, T=10.0, dt=0.05,
    n_trajectories=512, n_eval_steps=600, n_improve_steps=40,
    lr_value=3e-3, lr_policy=1e-3,
    hidden_dim=128, n_layers=3,
    n_collocation=512, n_outer=60, device=device,
)
solver_m = PolicyIteration(problem_m, cfg_m)
history_m = solver_m.solve(verbose=True)

xs_m = np.linspace(0.2, 10.0, 200)
V_exact_m = merton_analytical_value(xs_m, sol_m["A"], gamma, reward_scale)

xs_t_m = torch.from_numpy(xs_m).float().unsqueeze(-1).to(device)
with torch.no_grad():
    V_learned_m = solver_m.V_net(xs_t_m).cpu().numpy().flatten()
    a_learned_m = solver_m.policy_net(xs_t_m).cpu().numpy()
pi_learned = a_learned_m[:, 0]
cr_learned = a_learned_m[:, 1]

# --- Merton Fig 1: Value + Policy ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(xs_m, V_exact_m, "r--", lw=2, label=r"$V^*(x)$ exact")
axes[0].plot(xs_m, V_learned_m, "b-", lw=2, label=r"$V_\theta(x)$ learned")
axes[0].set_xlabel("Wealth $x$"); axes[0].set_ylabel("$V(x)$")
axes[0].set_title("Value Function"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(xs_m, pi_learned, "b-", lw=2, label=r"Learned $\pi(x)$")
axes[1].axhline(sol_m["pi_star"], color="r", ls="--", lw=2,
                label=rf"$\pi^* = {sol_m['pi_star']:.3f}$")
axes[1].set_xlabel("Wealth $x$"); axes[1].set_ylabel(r"Risky fraction $\pi$")
axes[1].set_title("Risky Fraction"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(xs_m, cr_learned, "b-", lw=2, label=r"Learned $c/x$")
axes[2].axhline(sol_m["k"], color="r", ls="--", lw=2,
                label=rf"$k^* = {sol_m['k']:.4f}$")
axes[2].set_xlabel("Wealth $x$"); axes[2].set_ylabel("Consumption rate $c/x$")
axes[2].set_title("Consumption Rate"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.suptitle("Merton Problem — Learned vs Exact", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "merton_value_policy.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved merton_value_policy.png")

# --- Merton Fig 2: Convergence ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(history_m["eval_loss"], "o-", color="tab:blue", ms=3, lw=1.5)
axes[0].set_ylabel("Value-fit MSE"); axes[0].set_xlabel("PI iteration")
axes[0].set_yscale("log"); axes[0].set_title("Value-fit MSE"); axes[0].grid(True, alpha=0.3)

axes[1].plot(history_m["improve_loss"], "o-", color="tab:orange", ms=3, lw=1.5)
axes[1].set_ylabel(r"$-\mathbb{E}[Q]$"); axes[1].set_xlabel("PI iteration")
axes[1].set_title("Policy Objective"); axes[1].grid(True, alpha=0.3)

axes[2].plot(history_m["hjb_residual"], "o-", color="tab:red", ms=3, lw=1.5)
axes[2].set_ylabel("HJB residual"); axes[2].set_xlabel("PI iteration")
axes[2].set_yscale("log"); axes[2].set_title("HJB Residual"); axes[2].grid(True, alpha=0.3)
plt.suptitle("Merton Problem — Convergence", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "merton_convergence.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved merton_convergence.png")

# --- Merton Fig 3: Sample wealth trajectories + cumulative reward ---
n_samples_m = 5
with torch.no_grad():
    x0_m = torch.full((n_samples_m, 1), 1.0, device=device)
    T_sim_m = 40.0
    dt_sim_m = 0.05
    ts_sim_m = torch.linspace(0, T_sim_m, int(T_sim_m / dt_sim_m) + 1, device=device)
    sde_m = ClosedLoopSDE(problem_m, solver_m.policy_net, problem_m.noise_type)
    traj_m = torchsde.sdeint(sde_m, x0_m, ts_sim_m, method="euler", dt=dt_sim_m)
    # traj_m: (steps+1, n_samples_m, 1)
    wealth_np = traj_m[:, :, 0].cpu().numpy()  # (steps+1, n_samples_m)
    ts_m_np = ts_sim_m.cpu().numpy()

    st_flat_m = traj_m[:-1].reshape(-1, 1)  # (n_steps*n_samples_m, 1)
    act_flat_m = solver_m.policy_net(st_flat_m)
    rew_flat_m = problem_m.reward(st_flat_m, act_flat_m).cpu().numpy().flatten()
    n_steps_m = len(ts_m_np) - 1
    rew_all_m = rew_flat_m.reshape(n_steps_m, n_samples_m)
    disc_m = np.exp(-rho_m * ts_m_np[:-1])
    cum_reward_m = np.cumsum(disc_m[:, None] * rew_all_m * dt_sim_m, axis=0)

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
colors_m = plt.cm.tab10(np.linspace(0, 1, n_samples_m))
for i in range(n_samples_m):
    axes[0].plot(ts_m_np, wealth_np[:, i], lw=1.2, color=colors_m[i], alpha=0.7)
axes[0].set_ylabel("Wealth $X_t$"); axes[0].set_title(f"Optimal Wealth Trajectories ($X_0=1$, {n_samples_m} samples)")
axes[0].grid(True, alpha=0.3)

for i in range(n_samples_m):
    axes[1].plot(ts_m_np[:-1], cum_reward_m[:, i], lw=1.2, color=colors_m[i], alpha=0.7)
axes[1].set_xlabel("Time $t$"); axes[1].set_ylabel("Cumulative discounted reward")
axes[1].set_title("Cumulative Discounted Reward"); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "merton_trajectory.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved merton_trajectory.png")

print("\n" + "="*60)
print("All figures saved to:", OUT_DIR)
print("="*60)
