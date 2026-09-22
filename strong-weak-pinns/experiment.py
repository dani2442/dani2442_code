"""Matched strong/weak PINNs on five one-dimensional PDEs, plus exact loss-Hessian conditioning."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
POST = HERE.parents[1] / "content/posts/strong-weak-pinns"
COUNTS = (4, 8, 16, 32, 64)
METHODS = ("strong", *(f"weak-{m}" for m in COUNTS))


def setup():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


@lru_cache(maxsize=None)
def quadrature(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (torch.tensor((nodes[:, None] + 1) / 2),
            torch.tensor(weights[:, None] / 2))


def exact(x, order=0):
    """The manufactured solution and its spatial derivatives."""
    freq = torch.tensor([1., 8.]) * torch.pi
    amplitude = torch.tensor([1., .1])
    return (amplitude * freq**order * torch.sin(x * freq + order * torch.pi / 2)).sum(1, keepdim=True)


def basis(x, m):
    """L2-orthonormal sine modes, their derivatives, and their energies lambda_k = (k pi)^2."""
    freq = torch.arange(1, m + 1) * torch.pi
    return (2**.5 * torch.sin(x * freq),
            2**.5 * freq * torch.cos(x * freq), freq.square())


def derivative(y, x, create_graph=True):
    return torch.autograd.grad(y.sum(), x, create_graph=create_graph)[0]


def primitive(fun, x, n=64):
    """F(x_i) = integral_0^{x_i} fun, column by column, by Gauss-Legendre on each [0, x_i]."""
    nodes, weights = quadrature(n)
    points = (x[:, None, :] * nodes[None, :, :]).reshape(-1, 1)
    values = fun(points).reshape(x.shape[0], n, -1)
    return x * (weights[None, :, :] * values).sum(1)


def dual_norm_squared(fun, x, w):
    """||fun||_{V'}^2 for V = H_0^1(0,1) with the energy norm: the variance of a primitive."""
    F = primitive(fun, x)
    F = F - (w * F).sum()
    return (w * F.square()).sum()


@dataclass(frozen=True)
class Problem:
    """A u = -(D u')' + a u' + c u^3 + beta u u' on (0,1), with D(x) = d0 + d1 x.

    Writing A u = -flux(u)' + source(u) with flux = D u' - beta u^2 / 2 and
    source = a u' + c u^3, the weak residual is int flux v' + source v - f v dx.
    """
    key: str
    title: str
    d0: float
    d1: float = 0.
    a: float = 0.
    c: float = 0.
    beta: float = 0.

    def diffusion(self, x):
        return self.d0 + self.d1 * x

    def flux(self, x, u, du):
        return self.diffusion(x) * du - self.beta / 2 * u.square()

    def source(self, u, du):
        return self.a * du + self.c * u**3

    def operator(self, x, u, du, ddu):
        return (-self.diffusion(x) * ddu - self.d1 * du + self.beta * u * du
                + self.source(u, du))

    def forcing(self, x):
        return self.operator(x, exact(x), exact(x, 1), exact(x, 2))

    def weak(self, x, w, u, du, phi, dphi):
        """The weak form tested against the columns of phi, without the load int f phi."""
        return dphi.T @ (w * self.flux(x, u, du)) + phi.T @ (w * self.source(u, du))

    def linearized(self, x, phi, dphi, lam):
        """Derivative at u* in the mode directions: (strong operator, flux part, source part)."""
        u, du = exact(x), exact(x, 1)
        dflux = self.diffusion(x) * dphi - self.beta * u * phi
        dsource = self.a * dphi + 3 * self.c * u.square() * phi
        strong = (self.diffusion(x) * lam * phi - self.d1 * dphi
                  + self.beta * (du * phi + u * dphi) + dsource)
        return strong, dflux, dsource


PROBLEMS = (Problem("poisson", "Poisson", 1.),
            Problem("advection", "Advection–diffusion", .1, a=1.),
            Problem("elasticity", "Elastic bar", 1., d1=1.),
            Problem("reaction", "Reaction–diffusion", 1., c=1.),
            Problem("burgers", "Steady Burgers", .1, beta=1.))


class PINN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, x):
        return x * (1 - x) * self.net(2 * x - 1)


class Objective:
    def __init__(self, problem, n):
        self.problem = problem
        self.x, self.w = quadrature(n)
        self.phi, self.dphi, self.lam = basis(self.x, max(COUNTS))
        self.f = problem.forcing(self.x)
        self.load = self.phi.T @ (self.w * self.f)
        self.strong_scale = (self.w * self.f.square()).sum()
        # One full forcing norm for ALL m, including tests that miss load modes.
        self.weak_scale = dual_norm_squared(problem.forcing, self.x, self.w)

    def __call__(self, model, method):
        x = self.x.detach().requires_grad_(True)
        u = model(x)
        du = derivative(u, x)
        if method == "strong":
            residual = self.problem.operator(x, u, du, derivative(du, x)) - self.f
            return (self.w * residual.square()).sum() / self.strong_scale
        m = int(method.split("-")[1])
        b = self.problem.weak(x, self.w, u, du, self.phi[:, :m], self.dphi[:, :m]) - self.load[:m]
        return (b.flatten().square() / self.lam[:m]).sum() / self.weak_scale


def measure(model, n=512):
    """Common relative H_0^1 error, evaluated independently of the training rule."""
    nodes, w = quadrature(n)
    x = nodes.detach().requires_grad_(True)
    du = derivative(model(x), x, create_graph=False)
    target = exact(x, 1)
    return torch.sqrt((w * (du - target).square()).sum() / (w * target.square()).sum()).item()


def hessians(problem, n, points=512):
    """Gauss-Newton Hessians of both full losses at u*, in L2-orthonormal sine coefficients."""
    x, w = quadrature(points)
    phi, dphi, lam = basis(x, n)
    strong, dflux, dsource = problem.linearized(x, phi, dphi, lam)
    # The linearized weak residual is int g v' with g = dflux - primitive(dsource),
    # and its dual norm is the L2 norm of g minus its mean.
    g = dflux - primitive(lambda t: problem.linearized(t, *basis(t, n))[2], x)
    g = g - (w * g).sum(0, keepdim=True)
    return strong.T @ (w * strong), g.T @ (w * g)


def condition_numbers(points=512):
    counts = np.array([2, *COUNTS])
    data = {"N": counts}
    for problem in PROBLEMS:
        values = []
        for n in counts:
            row = []
            for hessian in hessians(problem, int(n), points):
                eigenvalues = torch.linalg.eigvalsh(hessian)
                if eigenvalues[0] <= 0:
                    raise RuntimeError("The coefficient Hessian must be positive definite.")
                row.append((eigenvalues[-1] / eigenvalues[0]).item())
            values.append(row)
        data[problem.key] = np.array(values)
    return data


def validate():
    """Check the equations, dual norms, projections, linearizations, and the trained gradients."""
    setup()
    x0, w = quadrature(256)
    phi, dphi, lam = basis(x0, max(COUNTS))
    gram = dphi.T @ (w * dphi)
    torch.testing.assert_close(gram, torch.diag(lam), atol=1e-8, rtol=1e-10)
    torch.manual_seed(2026)
    model = PINN(4)
    trial, dtrial, tlam = basis(x0, 4)
    for problem in PROBLEMS:
        x = x0.detach().requires_grad_(True)
        u = exact(x)
        du = derivative(u, x)
        # The analytic derivatives in exact() agree with autograd.
        torch.testing.assert_close(problem.operator(x, u, du, derivative(du, x)),
                                   problem.forcing(x), atol=1e-10, rtol=1e-10)
        objective = Objective(problem, 256)
        assert objective(exact, "strong").item() < 1e-20
        assert objective(exact, "weak-64").item() < 1e-20
        # Integration by parts on a non-solution, including the nonlinear terms.
        u = model(x)
        du = derivative(u, x)
        residual = problem.operator(x, u, du, derivative(du, x)) - problem.forcing(x)
        weak = problem.weak(x, w, u, du, phi, dphi) - objective.load
        torch.testing.assert_close(weak, phi.T @ (w * residual), atol=1e-9, rtol=1e-9)
        # The linearization at u* matches central differences of the operator.
        strong, dflux, _ = problem.linearized(x0, trial, dtrial, tlam)
        us, dus, ddus = exact(x0), exact(x0, 1), exact(x0, 2)
        h = 1e-6
        central = (problem.operator(x0, us + h * trial, dus + h * dtrial, ddus - h * tlam * trial)
                   - problem.operator(x0, us - h * trial, dus - h * dtrial, ddus + h * tlam * trial)) / (2 * h)
        torch.testing.assert_close(central, strong, atol=1e-6, rtol=1e-6)
        # 64 tests almost see the full dual norm of a four-mode linearized residual.
        g = dflux - primitive(lambda t: problem.linearized(t, *basis(t, 4))[2], x0)
        b = dphi.T @ (w * g)
        projected = b.T @ (b / lam[:, None])
        full = hessians(problem, 4)[1]
        assert torch.linalg.eigvalsh(full - projected).min() > -1e-8
        torch.testing.assert_close(projected, full, atol=2e-3, rtol=1e-4)
        # Parameter gradients of the trained objectives against central differences.
        p = next(model.parameters())
        for method in ("strong", "weak-4", "weak-64"):
            model.zero_grad(set_to_none=True)
            objective(model, method).backward()
            analytic, original = p.grad[0, 0].item(), p[0, 0].item()
            values = []
            for delta in (1e-5, -1e-5):
                with torch.no_grad():
                    p[0, 0] = original + delta
                values.append(objective(model, method).item())
            with torch.no_grad():
                p[0, 0] = original
            np.testing.assert_allclose(analytic, (values[0] - values[1]) / 2e-5, rtol=1e-5, atol=1e-8)
    # Poisson: Riesz identity ||f||_{V'} = ||u*||_V, an unseen eighth mode, exact N^4 / N^2.
    poisson = Objective(PROBLEMS[0], 256)
    torch.testing.assert_close(poisson.weak_scale, (w * exact(x0, 1).square()).sum())
    de = .1 * 8 * torch.pi * torch.cos(8 * torch.pi * x0)
    b = dphi.T @ (w * de)
    assert b[:4].norm() < 1e-10
    energy = (w * de.square()).sum()
    torch.testing.assert_close((b.flatten().square() / lam).sum(), energy)
    change = torch.eye(64) + .01 * torch.randn(64, 64)
    transformed = change.T @ b
    torch.testing.assert_close((transformed.T @ torch.linalg.solve(change.T @ gram @ change, transformed)).squeeze(), energy)
    conditions, refined = condition_numbers(), condition_numbers(1024)
    for problem in PROBLEMS:
        np.testing.assert_allclose(conditions[problem.key], refined[problem.key], rtol=1e-7)
    np.testing.assert_allclose(conditions["poisson"],
                               np.column_stack([conditions["N"]**4, conditions["N"]**2]), rtol=1e-8)
    print("Validated all five PDEs, integration by parts, dual norms, linearizations, gradients, and Hessians.", flush=True)


def train(args):
    setup()
    args.results.mkdir(parents=True, exist_ok=True)
    histories, checks = {}, {}
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    device="cpu", dtype="float64", threads=1,
                    problems=[asdict(p) for p in PROBLEMS], methods=list(METHODS),
                    counts=list(COUNTS), comparison_tests=max(COUNTS),
                    history_columns=["step", "normalized_loss", "common_error"],
                    solution_modes=[1, 8], solution_amplitudes=[1., .1],
                    evaluation_points=512, common_error="relative H_0^1 seminorm",
                    weak_test_norm="H_0^1 seminorm", complete=False)
    metadata_path = args.results / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for problem in PROBLEMS:
        objective = Objective(problem, args.points)
        refined_objective = Objective(problem, 2 * args.points)
        for method in METHODS:
            for seed in args.seeds:
                torch.manual_seed(seed)
                model = PINN(args.width)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                history = []
                for step in range(args.steps + 1):
                    if step % args.log_every == 0 or step == args.steps:
                        history.append([step, objective(model, method).item(), measure(model)])
                    if step == args.steps:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective(model, method)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss: {problem.key}, {method}, seed {seed}, step {step}")
                    loss.backward()
                    optimizer.step()
                error, refined_error = history[-1][2], measure(model, 1024)
                error_gap = abs(error - refined_error) / max(refined_error, 1e-12)
                loss_gap = abs(history[-1][1] - refined_objective(model, method).item())
                if error_gap > 1e-5 or loss_gap > 1e-7:
                    raise RuntimeError(f"Quadrature check failed: {problem.key}, {method}, {seed}, {error_gap}, {loss_gap}")
                key = f"{problem.key}_{method}_seed{seed}"
                histories[key] = np.array(history)
                checks[key] = {"error_refinement_relative_gap": error_gap,
                               "loss_refinement_absolute_gap": loss_gap}
                # Incomplete datasets cannot be plotted as final results.
                np.savez_compressed(args.results / "histories.npz", **histories)
                (args.results / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
                print(f"{key}: normalized loss={history[-1][1]:.3g}, common error={error:.3g}", flush=True)
    np.savez_compressed(args.results / "conditioning.npz", **condition_numbers())
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--figures", type=Path, default=POST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.width, args.log_every) < 1 or args.points < 128 or args.lr <= 0:
        parser.error("Use positive steps, width, log interval and learning rate, and at least 128 quadrature points for 64 tests.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct.")
    if not args.plot_only:
        validate()
        if args.validate_only:
            return
        train(args)
    from plots import plot
    plot(args.results, args.figures)


if __name__ == "__main__":
    main()
