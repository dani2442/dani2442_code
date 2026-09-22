"""
Compliance minimization driven by the topological derivative.

The design is a material indicator  chi : cells -> {0, 1}  on the triangulated
design domain.  At every iteration we

  1. solve the elasticity problem on the current design (void cells carry the
     ersatz modulus E_min = 1e-6 E, a numerical stand-in for a traction-free
     hole);
  2. evaluate the topological derivative of the compliance at every *solid*
     point, D_T J = (1/E)[4 sigma:sigma - (tr sigma)^2] - the exact cost, per
     unit area, of nucleating an infinitesimal traction-free circular hole
     there;
  3. regularize it with a cone filter of radius r_min (this both fixes the
     minimum feature size and extends the field into the void, which is what
     lets material be re-inserted);
  4. keep the cells with the largest filtered gradient, up to the volume
     allowed by a shrinking schedule V_k.

Step 4 is the Lagrangian form of the volume constraint: for L = J + l |Omega|
the change under hole nucleation is |B|(D_T J - l), so cells with
D_T J < l should be voided, and the multiplier l is exactly the V_k-quantile of
the filtered gradient. Steps 3-4 follow the BESO update of Huang & Xie, using
recursive temporal averaging and the topological derivative in place of their
heuristic sensitivity.

This module is only the driver: which load cases to run, at which settings, and
which figures to make of them.  The pieces live in

    problem.py    the Problem base class - mesh, boundary data, drawing hooks
    examples/     one module per load case, each subclassing Problem
    solver.py     the cone filter, the state solve and the optimization loop
    draw.py       drawing primitives, including the Gamma_D/Gamma_N symbols
    figures.py    the figures of the post
    fem.py        P1 plane elasticity and the topological-derivative formula
    style.py      the shared palette and matplotlib settings

Outputs (written to ../../content/posts/topological_derivative/):
    td_mesh.png        the triangulation and boundary conditions
    td_gradient.png    the topological gradient on the full domain
    td_optimization.gif  actual designs, their deflection and the history
    td_convergence.png   static convergence history
    td_bridge.gif        three-pier bridge with piers and deck traction
    td_hanging.png       hanging bridge: deck underneath, two end piers
    td_hanging_sweep.png hanging designs across volume and filter radius
    td_suspended.png     the same deck hung from three piers on the top edge
    td_lbracket.png      L-bracket with clamp and traction
    td_bridge_sweep.png  bridge designs across volume and filter radius
    td_results.json      parameters and numerical histories behind the figures
"""

from pathlib import Path
import argparse
import json

import numpy as np

import examples
import figures
import problem
import solver
import style

OUT = Path(__file__).resolve().parents[2] / "content/posts/topological_derivative"

SWEEP_RADII = (2.5, 3.5, 5.5)
SWEEP_VOLUMES = (.30, .40, .50)
BASELINE = (3.5, .40)               # the sweep panel that gets its own figure


class Runner:
    """Runs load cases, optionally from a cache, and collects the summary.

    The cache stores the numerical result of a run keyed by everything that
    determines it, so figure layouts can be adjusted without solving again.
    """

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.summary = []

    def run(self, p, record=False):
        key = (f"{p.name}{p.cache_tag}_{p.nx}x{p.ny}"
               f"_v{p.vol_frac}_r{p.rmin_cells}_n{p.n_iter}_record{record}")
        cache = self.cache_dir / (key + ".npz") if self.cache_dir else None
        if cache and cache.exists():
            with np.load(cache) as data:
                hist = {k: data[k] for k in ("J", "V", "chi", "U") if k in data}
                chi = data["final"]
            print(f"loaded {key}")
        else:
            print(f"{key}:")
            chi, hist = solver.optimize(p, record=record)
            if cache:
                np.savez_compressed(cache, final=chi, **hist)
        self._check(p, chi, hist)
        self.summary.append(self._describe(p, hist))
        return chi, hist

    @staticmethod
    def _check(p, chi, hist):
        """The invariants the figures and the post's numbers rely on."""
        J, V = np.asarray(hist["J"]), np.asarray(hist["V"])
        assert np.isfinite(J).all() and (J > 0).all()
        assert abs(V[-1] - p.vol_frac) <= 1 / (~p.void).sum()
        assert np.all(chi[p.keep & ~p.void] == 1) and np.all(chi[p.void] == 0)

    @staticmethod
    def _describe(p, hist):
        """One record of td_results.json."""
        return dict(name=p.name, nx=p.nx, ny=p.ny, lx=p.lx, ly=p.ly,
                    target_volume=p.vol_frac, rmin_cells=p.rmin_cells,
                    iterations=p.n_iter, evolution_rate=p.evol_rate,
                    E=problem.E_SOLID, E_min=problem.E_MIN, nu=problem.NU,
                    model=problem.MODEL,
                    support_condition=p.support_condition or p.label,
                    total_force=p.f.reshape(-1, 2).sum(axis=0).tolist(),
                    J=np.asarray(hist["J"]).tolist(),
                    V=np.asarray(hist["V"]).tolist())


def run_cantilever(runner, out):
    """The introductory case: setup, raw gradient, animation and history."""
    figures.figure_mesh(examples.Cantilever(24, 12), out)
    p = examples.Cantilever()
    _, _, g_tri = solver.solve_state(p, np.ones(p.n_cells))
    figures.figure_gradient(p, g_tri, out)
    _, hist = runner.run(p, record=True)
    figures.figure_animation(p, hist, out)
    figures.figure_convergence(p, hist, out)


def run_sweep(runner, out, build, filename):
    """Nine runs of one load case over filter radius and material budget.

    The baseline panel also gets a figure of its own: an animation for the load
    cases that have one, otherwise a single annotated design.
    """
    results = []
    for radius in SWEEP_RADII:
        for volume in SWEEP_VOLUMES:
            p = build(vol_frac=volume, rmin_cells=radius)
            baseline = (radius, volume) == BASELINE
            animate = baseline and p.gif_file is not None
            chi, hist = runner.run(p, record=animate)
            results.append((p, chi, hist))
            if animate:
                figures.figure_animation(p, hist, out)
            elif baseline:
                figures.figure_example(p, chi, hist, out)
    figures.figure_sweep(results, out, filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--cache-dir", type=Path,
                        help="reuse saved numerical runs when adjusting figure layouts")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    style.use()
    runner = Runner(args.cache_dir)

    run_cantilever(runner, out)
    run_sweep(runner, out, examples.Bridge, "td_bridge_sweep.png")
    run_sweep(runner, out, examples.HangingBridge, "td_hanging_sweep.png")

    # The suspended deck is the exact vertical mirror of the three-pier
    # bridge, so one run at the baseline settings is enough: sweeping it would
    # reproduce td_bridge_sweep.png upside down.
    for build in (examples.SuspendedBridge, examples.LBracket):
        p = build()
        chi, hist = runner.run(p)
        figures.figure_example(p, chi, hist, out)

    (out / "td_results.json").write_text(json.dumps(runner.summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
