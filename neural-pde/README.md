# Neural ODEs and Neural PDEs — data fitting in pure PyTorch

Companion code for the post *"Neural ODEs and Neural PDEs"*. Both scripts fit a
neural network embedded in a differential equation to noisy data, using only
`torch` + `numpy` + `matplotlib` (no `torchdiffeq`/FEM libraries). Gradients are
obtained by differentiating through the unrolled integrator with autograd — the
*discrete adjoint* of the equations derived in the post.

## Files

- `neural_ode.py` — Neural ODE `y' = f_theta(y,t)`, explicit (forward) Euler in
  time. Ground truth is the nonlinear spiral of Chen et al. (2018). Learns the
  vector field `f_theta` (a tanh MLP) from a noisy trajectory and recovers the
  phase portrait. Training uses the standard mini-batching of short
  *sub-trajectories* and keeps the best full-trajectory checkpoint.

- `neural_pde.py` — Neural PDE `y_t + A y + d(x,t,y) = f_theta(x,t,y)` on `(0,1)`
  with homogeneous Neumann data (Tröltzsch eq. (5.8) with the control replaced by
  a network; here the known reaction `d = 0`). **Explicit Euler in time, P1
  finite elements in space** (mass/stiffness assembly with mass lumping). Ground
  truth is a Fisher–KPP source; learns `f_theta` and recovers the nonlinearity.
  Several initial conditions are simulated so the data sweeps a wide state range
  — needed for the reaction to be identifiable away from a single trajectory.

## Run

```bash
# from the repo's code/ directory (uses its uv venv)
.venv/bin/python neural-pde/neural_ode.py
.venv/bin/python neural-pde/neural_pde.py
```

Figures are written to `../../content/posts/neural-pde/`.

## Numerical notes

- **Stability.** Explicit Euler is conditionally stable. The ODE cubic dynamics
  need `dt ≈ 0.005`; the diffusion needs `dt < h²/(2ν)` (CFL). The scripts pick
  step counts that satisfy these.
- **Robust training.** The Neural ODE initializes the last layer to zero (so
  `f_theta ≈ 0` and the initial rollout cannot blow up), clips gradients, and
  trains on short sub-trajectories — far easier than one long stiff rollout.
- **Regularizer.** Both losses add `(gamma/2)|theta|^2` (Tikhonov), which makes
  the reduced objective coercive — see §2.3 of the post.
