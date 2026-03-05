"""Visualization utilities for continuous-time policy iteration."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def plot_convergence(history: dict[str, list[float]], save_path: str | None = None):
    """Three-panel plot: evaluation loss, policy-improvement objective, HJB residual."""
    keys = [k for k in ("eval_loss", "improve_loss", "hjb_residual") if k in history]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    labels = {
        "eval_loss": ("Value-fit MSE", "tab:blue"),
        "improve_loss": ("−E[Q] (policy obj)", "tab:orange"),
        "hjb_residual": ("HJB residual", "tab:red"),
    }
    for ax, key in zip(axes, keys):
        title, color = labels[key]
        ax.plot(history[key], "o-", color=color, markersize=3, lw=1.5)
        ax.set_xlabel("PI iteration")
        ax.set_ylabel(title)
        ax.set_title(title)
        if key != "improve_loss":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 1-D comparisons (value & policy)
# ---------------------------------------------------------------------------

def plot_value_comparison_1d(
    xs: np.ndarray,
    V_exact: np.ndarray,
    V_learned: np.ndarray,
    title: str = "Value Function",
    xlabel: str = "x",
    save_path: str | None = None,
):
    """Plot exact vs learned V(x) for a 1-D state."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(xs, V_exact, "r--", lw=2, label="V*(x) exact")
    ax1.plot(xs, V_learned, "b-", lw=2, label="V_θ(x) learned")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("V(x)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(xs, np.abs(V_exact - V_learned), "k-", lw=1.5)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("|V* − V_θ|")
    ax2.set_title("Absolute Error")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_policy_comparison_1d(
    xs: np.ndarray,
    a_exact: np.ndarray,
    a_learned: np.ndarray,
    title: str = "Policy",
    xlabel: str = "x",
    save_path: str | None = None,
):
    """Plot exact vs learned policy for a 1-D state."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, a_exact, "r--", lw=2, label="a*(x) exact")
    ax.plot(xs, a_learned, "b-", lw=2, label="α_φ(x) learned")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Action")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 2-D value contour
# ---------------------------------------------------------------------------

def plot_value_2d(
    V_fn,
    bounds: tuple[float, float] = (-3.0, 3.0),
    resolution: int = 80,
    title: str = "Learned V(x)",
    save_path: str | None = None,
):
    """Contour plot of V(x₁, x₂).

    V_fn: callable  (N, 2) ndarray → (N,) ndarray
    """
    g = np.linspace(bounds[0], bounds[1], resolution)
    X1, X2 = np.meshgrid(g, g)
    pts = np.stack([X1.flatten(), X2.flatten()], axis=1)
    Z = V_fn(pts).reshape(resolution, resolution)

    plt.figure(figsize=(7, 6))
    plt.contourf(X1, X2, Z, levels=50, cmap="viridis")
    plt.colorbar(label="V(x)")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Trajectory plots
# ---------------------------------------------------------------------------

def plot_trajectory_1d(
    ts: np.ndarray,
    xs: np.ndarray,
    actions: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    title: str = "Trajectory",
    save_path: str | None = None,
):
    """Time-series plot of a 1-D controlled trajectory."""
    n_panels = 1 + (actions is not None) + (rewards is not None)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    idx = 0
    axes[idx].plot(ts, xs, "b-", lw=1.5)
    axes[idx].set_ylabel("State x")
    axes[idx].set_title(title)
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    if actions is not None:
        axes[idx].plot(ts[: len(actions)], actions, "g-", lw=1.5)
        axes[idx].set_ylabel("Action a")
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    if rewards is not None:
        axes[idx].plot(ts[: len(rewards)], rewards, "r-", lw=1.5)
        axes[idx].set_ylabel("Reward r")
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time t")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trajectory_2d(
    states: np.ndarray,
    target: np.ndarray | None = None,
    title: str = "State Trajectory",
    save_path: str | None = None,
):
    """Plot trajectory in a 2-D state space."""
    plt.figure(figsize=(7, 6))
    plt.plot(states[:, 0], states[:, 1], "b-", alpha=0.7, lw=1.2)
    plt.scatter(*states[0], c="green", s=100, marker="o", label="Start", zorder=5)
    plt.scatter(*states[-1], c="red", s=100, marker="x", label="End", zorder=5)
    if target is not None:
        plt.scatter(*target, c="gold", s=150, marker="*", label="Target", zorder=5)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
