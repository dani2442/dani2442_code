"""Brownian motion in the bean, run until it leaves.

Plain Euler-Maruyama for `dX = dW` (generator (1/2)Delta), one shared time step
for every path, vectorized over the paths that are still alive.  A path is
declared out as soon as `phi(X) >= 0`; its exit point is the crossing of the
last segment with {phi = 0}, found by bisection, so the recorded exit location
sits on the boundary rather than just past it.

The one approximation that matters is that a path can leave and come back
inside a single step, which biases the exit distribution outwards by O(sqrt(dt))
-- see `validate.py`, which measures it against the finite-difference solution.
"""

import numpy as np


def sample_exits(bean, x0, n_paths, dt, rng, n_trace=0, max_steps=400_000):
    """Simulate `n_paths` Brownian paths from `x0` until they leave the bean.

    Returns the exit points (n_paths, 2), the exit times, and the full
    trajectories of the first `n_trace` paths (for drawing).
    """
    x0 = np.asarray(x0, dtype=float)
    X = np.tile(x0, (n_paths, 1))
    alive = np.arange(n_paths)
    exit_pt = np.full((n_paths, 2), np.nan)
    exit_t = np.full(n_paths, np.nan)
    traces = [[x0.copy()] for _ in range(n_trace)]
    sd = np.sqrt(dt)

    step = 0
    while alive.size and step < max_steps:
        step += 1
        Xa = X[alive]
        Xn = Xa + sd * rng.standard_normal((alive.size, 2))
        out = bean.phi(Xn) >= 0.0
        if out.any():
            idx = alive[out]
            xb = bean.crossing(Xa[out], Xn[out])
            exit_pt[idx] = xb
            exit_t[idx] = step * dt
            Xn[out] = xb
        X[alive] = Xn
        for i in alive[alive < n_trace]:
            traces[i].append(X[i].copy())
        alive = alive[~out]

    if alive.size:
        raise RuntimeError(f"{alive.size} paths still inside after {max_steps} steps")
    return exit_pt, exit_t, [np.array(t) for t in traces]
