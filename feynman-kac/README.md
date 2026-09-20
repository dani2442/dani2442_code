# Feynman-Kac on a bean-shaped domain

Companion code for the post *"The Feynman-Kac formula"*. It estimates the
solution of a Dirichlet problem at one point by simulating Brownian motion, and
checks the estimate against a deterministic solve of the same problem. Only
`numpy` + `scipy.sparse` + `matplotlib`; Pillow encodes the GIF.

The quantity everything turns on is, for `Omega` the bean and `tau` the first
time Brownian motion started at `x` leaves it,

```
u(x) = E_x[ g(B_tau) ]   solves   (1/2) Laplace(u) = 0 in Omega,  u = g on dOmega.
```

## Files

- `domain.py` — the geometry. The bean is the zero set of a smooth level
  function built from three disks: two lobes smoothly unioned (log-sum-exp
  `smin`), one disk smoothly subtracted to carve the concave side. Everything
  geometric goes through it: the inside test, the bisected crossing of a segment
  with `{phi = 0}`, the boundary datum `g`, and an arclength parametrization of
  the boundary curve (`Arclength`) used to bin exit locations.

- `walk.py` — Euler-Maruyama for `dX = dW`, vectorized over the paths still
  alive, each stopped the first time `phi(X) >= 0` and placed on the boundary by
  bisecting the last segment. The one approximation that matters is that a path
  can leave and return inside a single step, which biases the exit law outwards
  by `O(sqrt(dt))`.

- `reference.py` — the deterministic side. Laplace's equation on the same
  geometry with the Shortley-Weller stencil: a node whose neighbour falls
  outside uses the true distance to `{phi = 0}` along that grid line rather than
  the grid spacing, so the curved boundary is second order instead of
  staircased. Writing the discrete problem as `A u = -B g`, one transposed solve
  returns the **discrete harmonic measure** of a node, `w = -B^T A^-T e`: the
  exit distribution as a vector of weights, with no sampling at all. The same
  factorization gives `E_x[tau]` from `(1/2) Laplace(v) = -1`.

- `fem.py` — the picture's solver: a genuine P1 finite element solve on an
  unstructured triangulation of the same bean (boundary resampled at spacing
  `h`, hexagonal interior lattice, Delaunay, cut back to the boundary polygon,
  Laplacian-smoothed). Nothing in the post's numbers depends on it —
  `reference.py` remains the reference — but it gives the companion still its
  mesh, and it agrees with the finite differences at `x0` to `5e-4` at the
  drawn `h = 0.05` and to `2e-5` at `h = 0.0125`.

- `figures.py` / `main.py` — the five assets of the post: the opening animation
  of a single path leaving the bean (bare geometry, no data on the boundary),
  the harmonic extension with its level sets, the animation of the Monte Carlo,
  and the empirical exit distribution against the computed harmonic measure --
  plus `fk_bean_fem.png`, the deterministic answer to the opening animation:
  the same frame, figure size and dpi (so it is pixel-for-pixel the size of the
  GIF), with the field and its mesh in place of the path and bare of any
  chrome. `main.py` caches the simulation in `mc_cache.npz`; `--fresh` re-runs
  it. The opening path is its own one-path run at a coarser step (`WALK_DT`,
  `WALK_SEED`), picked for a trajectory that crosses the whole domain.

- `validate.py` — the checks, each against something that does not know about
  the thing it is checking.

## What the checks report

```
(A) finite differences against an exact harmonic extension
    n= 200  h=0.01126  max error 2.300e-06
    n= 400  h=0.00561  max error 5.902e-07   rate 1.96
    n= 800  h=0.00280  max error 1.488e-07   rate 1.99

(B) Monte Carlo at x0 = [-0.2989  0.1004],  u_FD = -0.367598
    dt=1.6e-04  u_MC=-0.377081  +-0.0168 (95%)   bias=-0.00948   sqrt(dt)=0.0126
    dt=4.0e-05  u_MC=-0.384301  +-0.0167 (95%)   bias=-0.01670   sqrt(dt)=0.0063
    dt=1.0e-05  u_MC=-0.379585  +-0.0168 (95%)   bias=-0.01199   sqrt(dt)=0.0032

(C) mean exit time   FD 0.14520   MC 0.14660 +-0.00424 (95%)

(D) exit distribution over 48 arclength bins:  total variation 0.0143   (sampling noise alone ~ 0.0195)
```

(A) is the second-order claim for the boundary treatment. In (B) the
discretization bias sits at or inside the sampling interval at every step
tried, so 4,000 paths cannot resolve a `sqrt(dt)` trend -- which is the honest
statement: at this sample size the bias is not the binding error. (C) checks the
occupation-measure half of the formula, (D) the exit-measure half.

## Running it

```
python main.py            # writes the figures into content/posts/feynman-kac/
python main.py --fresh    # re-simulates first
python validate.py        # ~3 minutes
```
