"""Checks on the two solvers, each against something that does not know about it.

(A) The finite-difference solver against an exact harmonic function, on the
    bean itself -- boundary data sampled from Re((x+iy)^3) + e^x cos y, whose
    harmonic extension is itself.  Second order in h is the claim.
(B) The Monte Carlo estimate against the finite-difference value, over a range
    of time steps, with the 95% interval alongside -- the residual is the
    O(sqrt(dt)) exit-time discretization bias, and it should shrink like sqrt(dt).
(C) The mean simulated exit time against the Poisson problem (1/2)Delta v = -1.
(D) The empirical exit distribution against the discrete harmonic measure that
    the transposed linear solve returns, in total variation over arclength bins.
"""

import numpy as np

from domain import Bean, Arclength
from reference import Reference
from walk import sample_exits

X0 = np.array([-0.30, 0.10])   # the point the post estimates


class Harmonic(Bean):
    def g(self, p):
        x, y = p[..., 0], p[..., 1]
        return x ** 3 - 3 * x * y ** 2 + np.exp(x) * np.cos(y)


def check_fd():
    print("(A) finite differences against an exact harmonic extension")
    prev = None
    for n in (200, 400, 800):
        r = Reference(Harmonic(), n=n)
        f = r.solve()
        err = np.nanmax(np.abs(f[r.inside] - Harmonic().g(r.P)[r.inside]))
        rate = "" if prev is None else f"   rate {np.log2(prev / err):.2f}"
        print(f"    n={n:4d}  h={r.h:.5f}  max error {err:.3e}{rate}")
        prev = err


def check_mc(n_paths=4000, seed=7):
    bean = Bean()
    ref = Reference(bean, n=700)
    ref.solve()
    u, x0 = ref.value(X0)
    print(f"\n(B) Monte Carlo at x0 = {np.round(x0, 4)},  u_FD = {u:+.6f}")
    for dt in (1.6e-4, 4e-5, 1e-5):
        rng = np.random.default_rng(seed)
        ep, _, _ = sample_exits(bean, x0, n_paths, dt, rng)
        v = bean.g(ep)
        se = v.std(ddof=1) / np.sqrt(n_paths)
        print(f"    dt={dt:.1e}  u_MC={v.mean():+.6f}  +-{1.96 * se:.4f} (95%)"
              f"   bias={v.mean() - u:+.5f}   sqrt(dt)={np.sqrt(dt):.4f}")
    return bean, ref, x0, u


def check_tau(bean, ref, x0, n_paths=4000, dt=1e-5, seed=11):
    ref.exit_time()
    k, _ = ref.node(x0)
    rng = np.random.default_rng(seed)
    _, et, _ = sample_exits(bean, x0, n_paths, dt, rng)
    se = et.std(ddof=1) / np.sqrt(n_paths)
    print(f"\n(C) mean exit time   FD {ref.tau[k]:.5f}   MC {et.mean():.5f}"
          f" +-{1.96 * se:.5f} (95%)")


def check_measure(bean, ref, x0, n_paths=20000, dt=1e-5, nbin=48, seed=3):
    w, _ = ref.weights(x0)
    al = Arclength(bean.boundary(900))
    edges = np.linspace(0, al.total, nbin + 1)
    p_fd, _ = np.histogram(al.project(ref.bnd_xy), bins=edges, weights=w)
    rng = np.random.default_rng(seed)
    ep, _, _ = sample_exits(bean, x0, n_paths, dt, rng)
    p_mc, _ = np.histogram(al.project(ep), bins=edges)
    p_mc = p_mc / p_mc.sum()
    tv = 0.5 * np.abs(p_fd - p_mc).sum()
    print(f"\n(D) exit distribution over {nbin} arclength bins:"
          f"  total variation {tv:.4f}"
          f"   (sampling noise alone ~ {0.5 * np.sqrt(2 / np.pi) * np.sqrt(nbin / n_paths):.4f})")


if __name__ == "__main__":
    check_fd()
    bean, ref, x0, u = check_mc()
    check_tau(bean, ref, x0)
    check_measure(bean, ref, x0)
