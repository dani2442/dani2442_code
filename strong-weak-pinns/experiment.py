"""Strong and finite-test weak PINNs for 1D Poisson; CPU, float64.

Run this file to train, save measurements, and generate the post's figures.
Use --plot-only to regenerate figures from the saved measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/strong-weak-pinns-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
POST = HERE.parents[1] / "content/posts/strong-weak-pinns"
METHODS = ["strong", "weak-2", "weak-4", "weak-8", "weak-16", "weak-32", "raw-16"]
COLORS = ["#244b78", "#c07c32", "#aa4f52", "#658249", "#128c88", "#7558a0", "#656565"]


def setup():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def quadrature(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (torch.tensor((nodes[:, None] + 1) / 2),
            torch.tensor(weights[:, None] / 2))


def exact(x, order=0):
    """Manufactured solution, its derivative, or the forcing -u''."""
    freq = torch.tensor([1., 8.]) * torch.pi
    amplitude = torch.tensor([1., .1])
    if order == 1:
        return (amplitude * freq * torch.cos(x * freq)).sum(1, keepdim=True)
    return (amplitude * freq ** order * torch.sin(x * freq)).sum(1, keepdim=True)


def basis(x, m):
    freq = torch.arange(1, m + 1) * torch.pi
    return (2 ** .5 * torch.sin(x * freq),
            2 ** .5 * freq * torch.cos(x * freq), freq.square())


def derivative(y, x, create_graph=True):
    return torch.autograd.grad(y.sum(), x, create_graph=create_graph)[0]


class PINN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, x):
        return x * (1 - x) * self.net(2 * x - 1)


class Objective:
    def __init__(self, n):
        self.x, self.w = quadrature(n)
        self.phi, self.dphi, self.lam = basis(self.x, 32)
        self.f = exact(self.x, 2)
        self.load = self.phi.T @ (self.w * self.f)
        self.strong_scale = (self.w * self.f.square()).sum()
        # Fixed forcing scale, shared by every weak test count.
        self.weak_scale = (self.load.flatten().square() / self.lam).sum()

    def __call__(self, model, method):
        x = self.x.detach().requires_grad_(True)
        u = model(x)
        du = derivative(u, x)
        if method == "strong":
            residual = -derivative(du, x) - self.f
            return (self.w * residual.square()).sum() / self.strong_scale
        m = int(method.split("-")[1])
        residual = (self.dphi[:, :m].T @ (self.w * du) - self.load[:m]).flatten()
        if method.startswith("raw"):
            return residual.square().sum() / self.strong_scale
        return (residual.square() / self.lam[:m]).sum() / self.weak_scale


def measure(model, n=512):
    x, w = quadrature(n)
    x.requires_grad_(True)
    u = model(x)
    du = derivative(u, x)
    ddu = derivative(du, x, create_graph=False)
    e, de, residual = u - exact(x), du - exact(x, 1), -ddu - exact(x, 2)
    errors = [e, de, residual]
    targets = [exact(x), exact(x, 1), exact(x, 2)]
    return [float(torch.sqrt((w * a.square()).sum() / (w * b.square()).sum()).detach())
            for a, b in zip(errors, targets)]


def validate():
    """Check PDE sign, integration by parts, Gram weighting, and a blind mode."""
    setup()
    x, w = quadrature(128)
    phi, dphi, lam = basis(x, 16)
    x.requires_grad_(True)
    u = exact(x)
    du = derivative(u, x)
    assert torch.max(torch.abs(-derivative(du, x) - exact(x, 2))) < 1e-10
    r = dphi.T @ (w * du) - phi.T @ (w * exact(x, 2))
    assert torch.max(torch.abs(r)) < 1e-10
    gram = dphi.T @ (w * dphi)
    torch.testing.assert_close(gram, torch.diag(lam), atol=1e-9, rtol=1e-10)
    # A known error is invisible to tests 1..4 but has positive energy.
    de = .1 * 8 * torch.pi * torch.cos(8 * torch.pi * x)
    r = dphi.T @ (w * de)
    assert r[:4].norm() < 1e-10
    energy = (w * de.square()).sum()
    torch.testing.assert_close((r.flatten().square() / lam).sum(), energy)
    # Invariance under a nonsingular, nonorthogonal change of basis.
    torch.manual_seed(2026)
    change = torch.eye(16) + .03 * torch.randn(16, 16)
    transformed = change.T @ r
    transformed_gram = change.T @ gram @ change
    torch.testing.assert_close((transformed.T @ torch.linalg.solve(transformed_gram, transformed)).squeeze(), energy)
    # Derivative of the actual training objective against finite differences.
    model = PINN(4)
    objective = Objective(64)
    p = next(model.parameters())
    for method in ("strong", "weak-16", "raw-16"):
        model.zero_grad(set_to_none=True)
        loss = objective(model, method)
        loss.backward()
        analytic = p.grad[0, 0].item()
        original = p[0, 0].item()
        vals = []
        for delta in (1e-5, -1e-5):
            with torch.no_grad():
                p[0, 0] = original + delta
            vals.append(objective(model, method).item())
        with torch.no_grad():
            p[0, 0] = original
        np.testing.assert_allclose(analytic, (vals[0] - vals[1]) / 2e-5, rtol=1e-5, atol=1e-8)
    print("Validated PDE sign, weak identity, Gram invariance, blind mode, and parameter gradients.", flush=True)


def train(args):
    setup()
    args.results.mkdir(parents=True, exist_ok=True)
    objective = Objective(args.points)
    histories, predictions, rows = {}, {}, []
    plot_x = torch.linspace(0, 1, 1001)[:, None]
    for method in METHODS:
        for seed in args.seeds:
            torch.manual_seed(seed)  # Identical initialization across methods.
            model = PINN(args.width)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            history = []
            training_seconds = 0.
            for step in range(args.steps + 1):
                if step % args.log_every == 0 or step == args.steps:
                    loss_value = objective(model, method).item()
                    metrics = measure(model)
                    history.append([step, training_seconds, loss_value, *metrics])
                if step == args.steps:
                    break
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model, method)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss: {method}, seed {seed}, step {step}")
                loss.backward()
                optimizer.step()
                training_seconds += time.perf_counter() - start
            # Independently double evaluation quadrature and training quadrature.
            refined = measure(model, 1024)
            refined_objective = Objective(2 * args.points)(model, method).item()
            metrics = history[-1][3:]
            disagreement = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(metrics, refined))
            loss_gap = abs(history[-1][2] - refined_objective)
            if disagreement > 1e-5 or loss_gap > 1e-7:
                raise RuntimeError(f"Quadrature check failed: {method}, {seed}, {disagreement}, {loss_gap}")
            key = f"{method}_seed{seed}"
            histories[key] = np.array(history)
            predictions[key] = model(plot_x).detach().numpy().flatten()
            row = dict(method=method, seed=seed, objective=history[-1][2],
                       relative_l2=metrics[0], relative_energy=metrics[1],
                       relative_residual=metrics[2], training_seconds=training_seconds,
                       evaluation_refinement_relative_gap=disagreement,
                       objective_refinement_absolute_gap=loss_gap)
            rows.append(row)
            print(f"{key}: objective={row['objective']:.3g}, L2={metrics[0]:.3g}, energy={metrics[1]:.3g}, residual={metrics[2]:.3g}, {training_seconds:.1f}s", flush=True)
            # Preserve completed runs if a later run is interrupted.
            np.savez_compressed(args.results / "histories.npz", **histories)
            np.savez_compressed(args.results / "predictions.npz", x=plot_x.numpy().flatten(), **predictions)
            with (args.results / "summary.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(python=platform.python_version(), torch=torch.__version__,
                    numpy=np.__version__, matplotlib=matplotlib.__version__,
                    platform=platform.platform(), device="cpu", dtype="float64", threads=1,
                    history_columns=["step", "training_seconds", "objective", "relative_l2", "relative_energy", "relative_residual"],
                    solution_modes=[1, 8], solution_amplitudes=[1., .1])
    (args.results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def plot(results, figures):
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "savefig.dpi": 180})
    data = np.load(results / "histories.npz")
    predictions = np.load(results / "predictions.npz")
    metadata = json.loads((results / "metadata.json").read_text())
    seeds = metadata["seeds"]
    def curves(ax, method, column, color, label=None, xcol=0):
        values = np.stack([data[f"{method}_seed{s}"] for s in seeds])
        y = values[:, :, column]
        x = np.median(values[:, :, xcol], axis=0)
        ax.plot(x, np.median(y, axis=0), color=color, label=label or method)
        ax.fill_between(x, y.min(0), y.max(0), color=color, alpha=.12)
        ax.set_yscale("log")
        ax.set_xlabel("Adam updates" if xcol == 0 else "Training time (s; excludes diagnostics)")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for method, color in zip(["strong", "weak-16", "weak-32", "raw-16"], [COLORS[0], COLORS[4], COLORS[5], COLORS[6]]):
        curves(axes[0], method, 2, color)
        curves(axes[1], method, 4, color)
        curves(axes[2], method, 4, color, xcol=1)
    for ax, title in zip(axes, ["Own normalized training objective", "Common relative energy error", "Common error versus compute"]):
        ax.set_title(title)
    axes[0].legend()
    fig.savefig(figures / "training.png")
    plt.close(fig)

    fig, grid = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes = grid.flatten()
    counts = [2, 4, 8, 16, 32]
    for col, label, color in [(2, "Own objective", "#888888"), (3, "Relative L2 error", COLORS[0]),
                              (4, "Relative energy error", COLORS[4]), (5, "Relative strong residual", COLORS[2])]:
        values = np.array([[data[f"weak-{m}_seed{s}"][-1, col] for s in seeds] for m in counts])
        med = np.median(values, axis=1)
        ax = axes[0] if col == 2 else axes[1]
        ax.errorbar(counts, med, yerr=[med - values.min(1), values.max(1) - med],
                         marker="o", capsize=3, color=color, label=label)
    for ax, title in zip(axes[:2], ["Own normalized training objective", "Independent final errors"]):
        ax.set(xscale="log", yscale="log", xlabel="Number of weak tests m", title=title)
        ax.set_xticks(counts, labels=counts)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.legend(fontsize=9)
    for method, color in zip(["weak-4", "weak-8", "weak-16", "weak-32"], [COLORS[2], COLORS[3], COLORS[4], COLORS[5]]):
        curves(axes[2], method, 4, color)
        # Show the same prespecified seed for every solution, never the best run.
        axes[3].plot(predictions["x"], predictions[f"{method}_seed{seeds[0]}"], color=color, label=method)
    axes[2].set_title("Common relative energy error")
    axes[2].legend(fontsize=9)
    x = predictions["x"]
    axes[3].plot(x, np.sin(np.pi*x) + .1*np.sin(8*np.pi*x), "k--", label="Exact", lw=1.5)
    axes[3].set(xlabel="x", ylabel="u(x)", title=f"Final solutions (seed {seeds[0]})")
    axes[3].legend(fontsize=9)
    fig.savefig(figures / "projection.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    modes = np.arange(1, 33, dtype=float)
    axes[0].loglog(modes, modes**4, color=COLORS[0], label="Strong / unweighted sine tests")
    axes[0].loglog(modes, modes**2, color=COLORS[4], label="Energy-dual weak")
    axes[0].set(xlabel="Mode k", ylabel="Curvature / first-mode curvature", title="Exact modal Hessian")
    axes[0].legend(fontsize=9)
    modes = np.arange(1, 17, dtype=float)
    initial = 1 / modes
    steps = np.arange(2001)
    for power, color, label in [(4, COLORS[0], "Strong"), (2, COLORS[4], "Full weak")]:
        weights = modes**power
        error = initial[None, :] * (1 - weights / weights.max())[None, :] ** steps[:, None]
        common = np.sqrt((error**2 * modes**2).sum(1) / (initial**2 * modes**2).sum())
        axes[1].semilogy(steps, common, color=color, label=label)
    axes[1].set(xlabel="Gradient descent updates", ylabel="Energy error / initial energy error", title="Linear trial space, N = 16")
    axes[1].legend()
    # Rescaling v_k by 1/k changes raw weights from k^4 to k^2.
    axes[2].loglog(np.arange(1, 17), np.arange(1, 17, dtype=float)**4, label="Raw, v_k = phi_k", color=COLORS[0])
    axes[2].loglog(np.arange(1, 17), np.arange(1, 17, dtype=float)**2, label="Raw, v_k = phi_k / k", color=COLORS[2])
    axes[2].loglog(np.arange(1, 17), np.arange(1, 17, dtype=float)**2, "--", label="Gram-weighted, either basis", color=COLORS[4])
    axes[2].set(xlabel="Mode k", ylabel="Curvature / first-mode curvature", title="Same test space, different scaling")
    axes[2].legend(fontsize=9)
    fig.savefig(figures / "conditioning.png")
    plt.close(fig)

    # Keep the article's numeric table reproducible from the saved runs.
    lines = ["| Objective | Own loss | Relative L2 | Relative energy | Relative residual | Training (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        last = np.array([data[f"{method}_seed{s}"][-1] for s in seeds])
        median = np.median(last, axis=0)
        fields = [f"{median[2]:.2e}", *[f"{100 * median[c]:.2f}%" for c in (3, 4, 5)], f"{median[1]:.2f}"]
        lines.append("| " + " | ".join([method, *fields]) + " |")
    (results / "table.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--figures", type=Path, default=POST)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.points, args.width, args.log_every) < 1 or args.lr <= 0:
        parser.error("Steps, quadrature points, width, log interval, and learning rate must be positive.")
    if not args.plot_only:
        validate()
        if args.validate_only:
            return
        train(args)
    plot(args.results, args.figures)


if __name__ == "__main__":
    main()
