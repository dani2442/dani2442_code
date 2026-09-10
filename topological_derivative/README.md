# The topological derivative — compliance minimization from scratch

Companion code for the post *"The topological derivative and compliance
minimization"*. Plane linear elasticity is discretized with P1 (constant-strain)
triangles on a triangulated design domain, and the topological derivative of the
compliance drives a hole-nucleating topology optimization. Only `numpy` +
`scipy.sparse` + `matplotlib` — no FEM library, no mesh generator, no
optimization package. Pillow encodes the optimization animation as a GIF.

The quantity everything turns on is, for a traction-free circular hole of radius
`eps` nucleated at `x` in a plane-elastic body,

```
J(Omega \ B_eps(x)) = J(Omega) + pi eps^2 D_T J(x) + o(eps^2),

D_T J(x) = (kappa + 1)/(8 mu) * [ 4 sigma:sigma - (tr sigma)^2 ],
```

with `kappa = (3-nu)/(1+nu)` in plane stress and `3-4nu` in plane strain. In
plane stress the prefactor collapses to `1/E`, in plane strain to `(1-nu^2)/E`.

## Files

- `fem.py` — the numerics core. Structured triangulations of a logical quad grid
  (`rect_mesh`, `quarter_annulus_mesh`, both with union-jack alternating
  diagonals so the mesh has no directional bias), CST assembly written so that
  `K(E) = sum_e E_e K0_e` is a single scaling of a precomputed value array,
  Dirichlet elimination, consistent edge tractions, element stress recovery,
  and `topological_derivative`.

- `validate.py` — verification of the formula against meshed geometry, with no
  reference to the derivation. A circular hole of radius `a` is actually meshed
  inside a quarter disk of radius `R`; the hole-free compliance is known exactly
  (the affine solution `u = eps_inf x` is in the P1 space), so the difference
  quotient `[J(Omega_a) - J(Omega_0)]/|B_a|` is a clean estimate of `D_T J`.
  Reports (A) the `a/R -> 0` limit for three remote stress states, against the
  closed-form Lamé value in the biaxial case; (B) the `h -> 0` limit; and (C)
  that the three algebraic forms of the formula used in the post agree to
  machine precision.

- `topopt.py` — compliance minimization for five load cases (cantilever,
  three-pier bridge, hanging bridge, suspended deck, L-bracket), a GIF of every
  cantilever and bridge iteration, separate annotated load-case figures, and
  two nine-run parameter comparisons.

- `style.py` — one documented palette and matplotlib settings shared by every
  figure.

## Run

Install `numpy`, `scipy`, `matplotlib`, and `pillow` in your Python environment.
With the repository's `code/.venv` environment, run:

```bash
# from code/topological_derivative/ (uses the code/ uv venv)
cd code/topological_derivative
../.venv/bin/python validate.py   #  ~1 min
../.venv/bin/python topopt.py --cache-dir /tmp/td-post-runs  # several minutes
```

Figures are written to `content/posts/topological_derivative/` in the repository.
`topopt.py` resolves that location from its own file, so it can also be run from
the repository root. `validate.py` still expects this working directory.
Use `--output-dir PATH` with `topopt.py` to write preview assets elsewhere.
The optional cache stores numerical runs as compressed NumPy arrays, allowing
figures to be redrawn without solving again. Use a fresh cache directory after
changing the solver, material, load, or update rule.

Generated optimization assets:

| File | Content |
|---|---|
| `td_mesh.png` | Cantilever mesh, clamped edge, and loaded patch. |
| `td_gradient.png` | Raw topological derivative on the full-material cantilever, available separately from the animation. |
| `td_optimization.gif` | 91 solved states (iterations 0–90), each showing material, the filtered update score, and stiffness/volume history. |
| `td_convergence.png` | Static cantilever history. |
| `td_bridge.gif` | 121 solved states (iterations 0-120) of the baseline bridge: three pinned piers, uniform traction on the whole top edge. |
| `td_hanging.png` | Baseline hanging bridge: two end piers, uniform traction on the whole bottom edge. |
| `td_hanging_sweep.png` | Nine hanging-bridge configurations, same grid of `V` and `rmin/h`. |
| `td_suspended.png` | The bottom-edge load hung from three piers on the *top* edge — the exact vertical mirror of the `td_bridge` design. |
| `td_lbracket.png` | L-bracket with its top clamp and the downward traction on the upper face of the horizontal arm. |
| `td_bridge_sweep.png` | Nine bridge configurations: columns vary `V = 0.30, 0.40, 0.50`; rows vary `rmin/h = 2.5, 3.5, 5.5`. |
| `td_results.json` | Parameters, total force, and full compliance/volume histories for all twenty-one runs. |

Both GIFs use recorded solver states, including a fresh state solve after the
last update. The sensitivity shown is the cone-filtered, recursively averaged
score used by the update, including its extension into void cells. Each GIF has
one colour scale of its own, gamma-compressed and clipped at the 98th percentile
of the positive scores across that run. They animate material changes on a
fixed mesh.

Every bridge run uses the same `3 x 1` domain on a `180 x 60` cell mesh, three
pins imposing `ux = uy = 0` at `x = 0, 1.5, 3` on the bottom edge, and a uniform
downward traction of total magnitude 1 spread over the *whole* top edge
(`Gamma_N = {y = 1}`, `g = (0, -1/3)`), plus plane stress, `E = 1`, `nu = 0.3`,
`E_min = 1e-6`, evolution rate `0.02`, and 120 updates. The baseline is
`V = 0.40`, `rmin/h = 3.5`; all ratios use the same full-material compliance
`J0 = 1.71216` reported by the current run. Two cell rows of deck and the three
pier heads are protected (396 cells, 3.7% of the domain), so the traction and
the reactions always act on solid material. Every pier carries horizontal
reaction, which is what allows an arch to form; the discrete point constraints
are distinct from the continuum clamped-boundary assumptions in the post's
theorem.

The hanging-bridge runs share the domain, mesh, material, schedule and total
load, and move both boundaries to the bottom: `Gamma_N = {y = 0}` with the same
`g = (0, -1/3)`, and only two pins, at `x = 0, 3`. Its full-material compliance
is `J0 = 4.23710`. Two deck rows and the two pier heads are protected (366
cells). The corner nodes are both loaded and pinned, so `hx/lx = 0.56%` of the
nodal load is taken directly by the supports and does no work; all ratios for
this case use its own `J0`. With the load underneath, the optimum inverts: an
arch above a deck that acts as a partial tie, instead of a deck resting on
arches.

`suspended_bridge` keeps the bottom-edge load and puts three pins back, this
time on the top edge at `x = 0, 1.5, 3`. It is the exact vertical mirror of
`bridge`: reflecting `y -> ly - y` flips the parity of `i + j`, which is
precisely what maps a `/` union-jack diagonal onto the `\` its mirror needs,
so the reflected triangulation is the original one. The reflected load is the
original with `g` reversed, and both `J = f.u` and
`D_T J ~ 4 sigma:sigma - (tr sigma)^2` are even in the sign of the state. The
run confirms it: the two compliance histories agree to `1.3e-12` over all 121
iterations and the final designs differ in 0 of 10800 cells from an exact
reflection. One run at the baseline settings is therefore enough — a sweep
would reproduce `td_bridge_sweep.png` upside down. The practical consequence is
that this objective cannot tell a compression arch from a tension cable.

## Verification results

`validate.py` at `E = 1`, `nu = 0.3`, plane stress. Entries are
`numerical D_T J / formula`; the residual is the `O(a^2/R^2)` effect of a finite
outer radius, not an error in the formula.

| remote stress | formula | a/R=0.2 | a/R=0.1 | a/R=0.05 | a/R=0.025 |
|---|---|---|---|---|---|
| uniaxial `(1,0)`  | 3 | 1.13179 | 1.03047 | 1.00731 | 1.00165 |
| biaxial `(1,1)`   | 4 | 1.04158 | 1.01002 | 1.00242 | 1.00054 |
| shear `(1,-1)`    | 8 | 1.17689 | 1.04070 | 1.00976 | 1.00220 |
| *exact, biaxial*  | — | 1.04167 | 1.01010 | 1.00251 | 1.00063 |

The last row is the closed-form Lamé value `1/(1 - a^2/R^2)` for the biaxial
case: the finite element result tracks it to four or five digits, so the residual
really is geometry and not discretization. The errors fall by a factor of four
per halving of `a/R`, i.e. `O(a^2/R^2)` as the theory predicts, and the
`h -> 0` table falls by a factor of four per mesh refinement, i.e. `O(h^2)` —
which is the expected rate for the compliance, since
`J - J_h = ||u - u_h||_a^2`.

## What is rigorous and what is not

**Rigorous.** The formula itself, including its constant: `validate.py` confirms
it against meshed holes for both plane models, three remote stress states and
several Poisson ratios. The exact difference identity it comes from,
`J(Omega_eps) - J(Omega) = int_{dB_eps} (sigma(u) n) . u_eps ds`, is Betti
reciprocity with no asymptotics in it. The compliance is monotone under material
removal, and the bracket `4 sigma:sigma - (tr sigma)^2 = 3(s_I^2 + s_II^2) -
2 s_I s_II` is positive definite, so the two agree in sign.

**Not rigorous, and deliberately so.**

- *Ersatz material.* Void cells carry `E_min = 1e-6 E` rather than being removed
  from the mesh, so the "traction-free hole" is approximated. Re-solving the
  final cantilever design over `E_min/E` in `{1e-4, 1e-6, 1e-8, 1e-10}` moves
  the compliance by `3.3e-4`, `3.3e-6`, `3.3e-8`, `0` relative to the
  `E_min -> 0` limit — linear in `E_min/E`, three parts per million at the value
  used. The benefit is a fixed matrix pattern and no floating-node handling.
- *Insertion.* The formula prices hole *nucleation* in solid material. Putting
  material back is priced here by letting the cone filter carry solid-phase
  values into the void — a practical device, not a derived quantity. Amstutz
  (2006) gives the rigorous two-phase counterpart for an inclusion of finite
  contrast.
- *The update itself.* Thresholding the filtered gradient at the volume quantile,
  with temporal sensitivity averaging, follows the BESO update of Huang
  & Xie. Here the previously averaged score is stored, so the smoothing is
  recursive. It is a greedy descent on the topological optimality condition, not a
  convergent algorithm, and the continuum problem has no minimizer without the
  perimeter- or filter-type regularization that the cone filter supplies.

## Numerical notes

- **Grading.** `quarter_annulus_mesh` places radial nodes geometrically,
  `r_i = a (R/a)^(i/nr)`, and `validate.py` picks `nr` so cells stay nearly
  square. Without grading, the `1/r^2` stress concentration that the topological
  derivative measures is not resolved and the difference quotient is useless.
- **Union-jack diagonals.** A single fixed diagonal biases both the CST stress
  and the resulting topology along that direction; alternating it removes the
  bias. The residual diamond texture visible in the gradient figure is the
  piecewise-constant CST stress, not noise.
- **Cone filter radius.** `rmin_cells = 3.5` element sizes. Smaller radii give
  thinner members and more mesh dependence; larger radii wash out the members.
- **Protected cells.** The cells under the load patch (and at the point supports
  of either bridge) are pinned solid. Without that, the greedy step can void the
  material the load is applied to.
