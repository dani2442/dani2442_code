# Strong and weak PINNs

Code for `content/posts/strong-weak-pinns/index.md`: matched strong and weak
PINNs on five one-dimensional problems, plus the exact condition numbers of
both loss Hessians. The default run writes only `results/` and the post's
three figures.

## Run

From the repository root, in an environment with NumPy, Matplotlib, and CPU
PyTorch (Python 3.11 or later):

```bash
python -m venv /tmp/strong-weak-pinns-venv
/tmp/strong-weak-pinns-venv/bin/pip install numpy matplotlib
/tmp/strong-weak-pinns-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

python code/strong-weak-pinns/experiment.py                  # validate, train, plot
python code/strong-weak-pinns/experiment.py --plot-only      # figures from saved results
python code/strong-weak-pinns/experiment.py --validate-only  # analytic checks only
python code/strong-weak-pinns/experiment.py --steps 20 --seeds 0 \
  --results /tmp/pinns-smoke/results --figures /tmp/pinns-smoke/figures
```

Library versions and every setting are recorded in `results/metadata.json`.
Runs are float64, single-threaded, and deterministic for a given platform.
The default 90 runs take about a quarter of an hour on one CPU core.

## What is compared

All problems live on $(0,1)$ with $u(0)=u(1)=0$ and the manufactured solution
$u^\star = \sin(\pi x) + 0.1\sin(8\pi x)$, from which $f = Au^\star$ is computed.
Every operator has the form $Au = -(Du')' + au' + cu^3 + \beta uu'$:

| Key | Problem | Operator |
| --- | --- | --- |
| `poisson` | Poisson | $-u''$ |
| `advection` | Advection–diffusion | $-0.1\,u'' + u'$ |
| `elasticity` | Elastic bar, stiffness $E(x)=1+x$ | $-(E u')'$ |
| `reaction` | Reaction–diffusion | $-u'' + u^3$ |
| `burgers` | Steady Burgers | $-0.1\,u'' + uu'$ |

Two losses are trained, both normalized to be dimensionless:

- `strong`: $\mathcal L_s = \|Au_\theta - f\|_{L^2}^2 / \|f\|_{L^2}^2$, using second derivatives.
- `weak-m` for $m\in\{4,8,16,32,64\}$: $\mathcal L_w = \sum_{k\le m}\mathcal R_\theta(\phi_k)^2/\lambda_k$
  divided by $\|f\|_{V'}^2$, with sine tests $\phi_k=\sqrt2\sin(k\pi x)$,
  $\lambda_k=(k\pi)^2$, and $V=H_0^1$. This is the dual norm of the weak
  residual restricted to the span of the tests; it uses first derivatives only.
  The weak residual is $\int (Du_\theta' - \tfrac\beta2 u_\theta^2)\,v' + (au_\theta' + cu_\theta^3)\,v - fv$,
  so the Burgers term is integrated by parts in conservative form.

The forcing norm $\|f\|_{V'}$ is the $L^2$ norm of the mean-free primitive of
$f$, computed by nested Gauss–Legendre quadrature. The common error is
$\mathcal L=\|u_\theta'-u^{\star\prime}\|_{L^2}/\|u^{\star\prime}\|_{L^2}$ on an
independent 512-point rule. Training never sees $u^\star$.

Every run uses $u_\theta(x)=x(1-x)N_\theta(2x-1)$ with a 1–32–32–1 tanh
network, 256-point Gauss–Legendre quadrature, full-batch Adam at `1e-3`, and
3,000 steps. Seeds 0, 1, 2 give identical initial parameters to both losses.
Final iterates are reported; nothing is selected against the exact solution.

Condition numbers are those of the Gauss–Newton Hessians of both full losses at
$u^\star$ in the first $N$ sine coefficients, i.e. of the linearized operator;
for the linear problems this is the exact Hessian.

## Outputs

- `results/histories.npz`: key `<problem>_<method>_seed<seed>`, columns
  `step`, `normalized_loss`, `common_error`, recorded every 100 steps.
- `results/conditioning.npz`: condition numbers of the strong and weak loss
  Hessians in the first `N` sine coefficients, per problem.
- `results/checks.json`: quadrature-refinement gaps for every run.
- `results/metadata.json`: settings, versions, and `complete: true` once done.
- `training.png`, `conditioning.png`, `projection.png` in the post directory.

## Validation

`validate()` runs before training and asserts, to `1e-8` or tighter unless noted:

- the analytic derivatives of the manufactured solution agree with autograd,
  and both losses vanish at $u^\star$ for every problem;
- integration by parts, including the nonlinear terms, holds on the quadrature
  rule for a non-solution;
- the linearized operator matches central differences of the operator;
- 64 sine tests reproduce the full dual-norm Hessian of a four-mode error;
- parameter gradients of the trained objectives match central differences (`1e-5`);
- for Poisson: the Riesz identity $\|f\|_{V'}=\|u^\star\|_V$, the sine Gram
  matrix $\operatorname{diag}(\lambda_k)$, invariance of the weak loss under a
  random change of test basis, an error in mode 8 that is invisible to the first
  four tests yet has its full energy, and condition numbers exactly $N^4$ and $N^2$.

After training, each model is re-evaluated with doubled diagnostic and training
quadrature; the run fails if the common error moves by more than `1e-5`
relatively or the loss by more than `1e-7` absolutely.
