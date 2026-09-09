# The topological derivative — compliance minimization from scratch

Companion code for the post *"The topological derivative and compliance
minimization"*. Plane linear elasticity is discretized with P1 (constant-strain)
triangles on a triangulated design domain, and the topological derivative of the
compliance drives a hole-nucleating topology optimization. Only `numpy` +
`scipy.sparse` + `matplotlib` — no FEM library, no mesh generator, no
optimization package.

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

- `topopt.py` — compliance minimization for three load cases (cantilever, MBB
  beam, L-bracket) and all four figures of the post.

- `style.py` — one documented palette and matplotlib settings shared by every
  figure.

## Run

Both scripts resolve their output path relative to the working directory, so
run them from **inside** this directory:

```bash
# from code/topological_derivative/ (uses the code/ uv venv)
cd code/topological_derivative
../.venv/bin/python validate.py   #  ~1 min
../.venv/bin/python topopt.py     #  ~4 min
```

Figures are written to `../../content/posts/topological_derivative/`.

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
  with the sensitivity averaged over two iterations, is the BESO update of Huang
  & Xie. It is a greedy descent on the topological optimality condition, not a
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
  of the MBB beam) are pinned solid. Without that, the greedy step can void the
  material the load is applied to.
