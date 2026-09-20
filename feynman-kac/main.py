"""Build every asset of the post.

    python main.py [--out DIR] [--fresh]

The simulation is cached next to this file, so re-rendering the figures does not
re-run it; `--fresh` forces a new one.
"""

import argparse
import pathlib
import time

import numpy as np

import fem
import figures
import style
from domain import Bean
from reference import Reference
from walk import sample_exits

X0 = np.array([-0.30, 0.10])   # snapped to the nearest grid node below
N_PATHS = 6000
DT = 2e-5
N_TRACE = 14
SEED = 20250918

WALK_DT = 5e-5                 # the single path of the opening figure
WALK_SEED = 4                  # picked for a path that crosses the whole bean

FEM_H = 0.05                   # mesh size of the companion FEM picture: coarse
                               # enough that the cells are visible at 823 px

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "mc_cache.npz"
DEFAULT_OUT = HERE / ".." / ".." / "content" / "posts" / "feynman-kac"


def simulate(bean, x0, fresh=False):
    key = np.array([x0[0], x0[1], N_PATHS, DT, N_TRACE, SEED])
    if CACHE.exists() and not fresh:
        z = np.load(CACHE, allow_pickle=True)
        if np.array_equal(z["key"], key):
            print(f"reusing {CACHE.name}")
            return z["exits"], z["times"], list(z["traces"])
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    exits, times, traces = sample_exits(bean, x0, N_PATHS, DT, rng,
                                        n_trace=N_TRACE)
    print(f"simulated {N_PATHS} paths at dt={DT:g} in {time.time() - t0:.1f}s "
          f"(mean exit time {times.mean():.4f})")
    np.savez(CACHE, key=key, exits=exits, times=times,
             traces=np.array(traces, dtype=object))
    return exits, times, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    style.use()
    bean = Bean()
    ref = Reference(bean, n=700)
    ref.solve()
    u0, x0 = ref.value(X0)
    print(f"x0 = {np.round(x0, 5)}   u(x0) = {u0:+.6f}   (h = {ref.h:.5f})")

    exits, times, traces = simulate(bean, x0, fresh=args.fresh)
    v = bean.g(exits)
    print(f"MC  {v.mean():+.6f} +- {1.96 * v.std(ddof=1) / np.sqrt(len(v)):.4f}"
          f"   mean exit time {times.mean():.4f}")

    rng = np.random.default_rng(WALK_SEED)
    wp, wt, wtr = sample_exits(bean, X0, 1, WALK_DT, rng, n_trace=1)
    print(f"opening walk: tau = {wt[0]:.4f}  ({len(wtr[0])} steps)")
    figures.figure_walk(bean, X0, wtr[0], wp[0], out)

    pts, tri, u_fem, n_bnd, _ = fem.solve(bean, FEM_H)
    print(f"FEM h={FEM_H:g}: {len(pts)} nodes ({n_bnd} on the boundary), "
          f"{len(tri)} triangles,  u_FEM(x0) = "
          f"{fem.value_at(pts, tri, u_fem, x0):+.6f}  (FD {u0:+.6f})")
    figures.figure_fem(bean, pts, tri, u_fem, X0, out)

    figures.figure_field(bean, ref, x0, u0, out)
    figures.figure_measure(bean, ref, x0, exits, out)
    figures.figure_animation(bean, x0, u0, exits, traces, out)


if __name__ == "__main__":
    main()
