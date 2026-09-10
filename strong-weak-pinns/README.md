# Strong and weak PINNs

Reproducible experiments for `content/posts/strong-weak-pinns/index.md`.
Only this directory and that post's figures are written by the default run.

## Setup and run

From the repository root, create a separate environment (Python 3.11 or later):

```bash
python -m venv /tmp/strong-weak-pinns-venv
/tmp/strong-weak-pinns-venv/bin/python -m pip install numpy matplotlib
/tmp/strong-weak-pinns-venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/strong-weak-pinns-venv/bin/python code/strong-weak-pinns/experiment.py
```

Alternatively use the repository's `code/pyproject.toml` environment. Published
measurements used Python 3.13.11, NumPy 2.5.2, Matplotlib 3.11.1, and PyTorch
2.14.0+cpu. Full run settings and platform are recorded in `results/metadata.json`.
Seeds, float64, deterministic PyTorch algorithms, and one CPU thread are fixed;
different library versions or platforms can still change optimization trajectories.

The default is 21 runs: seven methods, three paired seeds, 3,000 Adam updates
each. Optimization took about two minutes total on the recorded machine;
independent quadrature diagnostics add overhead. Final iterates are reported,
without selecting checkpoints or tuning hyperparameters against the exact solution.

## What is compared

- `strong`: squared strong residual, using second spatial derivatives.
- `weak-2`, `weak-4`, `weak-8`, `weak-16`, `weak-32`: finite energy-dual residual
  norms, using first spatial derivatives and diagonal sine-test Gram weighting.
- `raw-16`: the same 16 sine tests without Gram weighting.

All methods use `u(x) = x(1-x) N(2x-1)`, a 1–32–32–1 tanh network, exact
homogeneous Dirichlet boundary conditions, 128-point Gauss–Legendre quadrature,
and Adam at `1e-3`. The manufactured solution is `sin(pi*x) + 0.1*sin(8*pi*x)`.
Training uses only the forcing; the exact solution supplies diagnostics.

Strong and raw losses divide by the squared L2 norm of the forcing. Every
weighted weak loss divides by the same squared energy-dual norm of the forcing,
computed from its first 32 coefficients (its only nonzero modes are 1 and 8).
Own loss values are not interchangeable accuracy measures. We also report
relative L2 solution error, relative energy error, and relative L2 strong residual.

## Outputs

- `results/summary.csv`: one row per method and seed, including final objectives,
  common errors, training time, and quadrature refinement discrepancies.
- `results/histories.npz`: key `<method>_seed<seed>`, with columns listed in
  `metadata.json`. Metrics are evaluated at update 0, every 100 updates, and the
  final update, on independent 512-point quadrature.
- `results/predictions.npz`: plotting grid `x` and final solutions under the same
  keys. Full model checkpoints are not stored.
- `results/table.md`: median summary table generated from the histories, used
  in the article. If retraining, update the article's table and interpretation too.
- `conditioning.png`: exact coefficient-space Hessian and gradient descent
  calculations, plus a test-basis scaling ablation; these are not neural runs.
- `training.png`: own objectives and common energy errors versus updates/time.
- `projection.png`: test-count ablation and seed-0 solution curves.

Figures go directly into `content/posts/strong-weak-pinns/`. Curves show medians
with min–max shading across seeds. Timing includes forward/backward passes and
Adam updates, but excludes diagnostics, setup, and plotting. Time-axis positions
are median cumulative times at each update; this is not a separate fixed-time
budget experiment.

## Validation and alternate runs

```bash
# Check analytic identities and finite-difference parameter gradients only.
python code/strong-weak-pinns/experiment.py --validate-only

# Replot the saved measurements without training.
python code/strong-weak-pinns/experiment.py --plot-only

# A small end-to-end run, preserving the published results and figures.
python code/strong-weak-pinns/experiment.py --steps 20 --seeds 0 \
  --results /tmp/pinns-smoke/results --figures /tmp/pinns-smoke/figures

# A longer independent experiment.
python code/strong-weak-pinns/experiment.py --steps 10000 --points 256 \
  --results /tmp/pinns-long/results --figures /tmp/pinns-long/figures
```

Validation checks the PDE sign, integration by parts, sine-test Gram matrix,
basis invariance, a blind high-frequency error, and parameter gradients of all
three loss types. Each trained model is checked with 1,024 diagnostic points
and twice its training quadrature. The run fails if diagnostic errors change
by more than `1e-5` relatively or its normalized objective by more than `1e-7`
absolutely. For very small projected losses, absolute discrepancies are the
useful check; a tiny tested loss alone is not a convergence certificate.

The experiment does not establish a neural inf-sup bound, measure parameter
Hessians, or demonstrate universal optimizer superiority. Covering the forcing's
modes does not cover all errors a neural network can generate.
