"""
Numerical verification of the topological-derivative formula.

The formula
    D_T J = k(E, nu) [ 4 sigma:sigma - (tr sigma)^2 ],   k = 1/E (plane stress)
is derived in the post from Betti's identity plus the exterior Kirsch field.
Here we check it against the finite-element compliance of a domain in which a
circular hole is *actually* meshed, with no reference to the derivation.

Test problem (a quarter of a disk, exploiting the symmetry of the load):
    Omega_a = { a <= r <= R, 0 <= theta <= pi/2 },
    u_y = 0 on theta = 0,   u_x = 0 on theta = pi/2      (symmetry planes)
    sigma n = sigma_inf n    on r = R                    (loaded outer arc)
    sigma n = 0              on r = a                    (traction-free hole)
For a *diagonal* remote stress sigma_inf the two symmetry planes are traction
free in the mixed sense, and the hole-free problem (a = 0) has the exact affine
solution u = eps_inf x, which P1 elements reproduce exactly.  Therefore

    J(Omega_0) = sigma_inf : eps_inf * |Omega_0|          (exact, no mesh error)

and the whole discretization error sits in J(Omega_a).  Dividing the difference
by the hole area gives a mesh- and radius-dependent estimate of D_T J that must
converge to the formula as h -> 0 and a/R -> 0.

Three remote states are used, which together pin down *both* invariants:
    (1, 0)  uniaxial,   (1, 1)  hydrostatic,   (1, -1)  pure shear.
"""

import numpy as np

import fem
import style

E0, NU = 1.0, 0.3
MODEL = "plane_stress"
R_OUT = 1.0


def _quarter_annulus_compliance(a, R, nt, sigma_inf):
    """FEM compliance of the quarter annulus, and the polygonal hole area."""
    # Geometric radial grading with nearly square cells: (R/a)^(1/nr) - 1 = dtheta
    dth = 0.5 * np.pi / nt
    nr = max(4, int(np.ceil(np.log(R / a) / np.log1p(dth))))
    nodes, tris, _, (ni, nj) = fem.quarter_annulus_mesh(a, R, nr, nt)
    pb = fem.Elasticity2D(nodes, tris, nu=NU, model=MODEL)
    nid = np.arange(len(nodes)).reshape(ni + 1, nj + 1)

    # Outer arc: consistent traction sigma_inf . n on each straight edge.
    edges = np.column_stack([nid[ni, :-1], nid[ni, 1:]])
    tau = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    nrm = np.column_stack([tau[:, 1], -tau[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    S = np.array([[sigma_inf[0], sigma_inf[2]], [sigma_inf[2], sigma_inf[1]]])
    f = pb.edge_load(edges, nrm @ S.T)

    fixed = np.concatenate([2 * nid[:, 0] + 1, 2 * nid[:, nj]])
    _, J = pb.solve(np.full(pb.n_tris, E0), f, fixed)

    area_hole = 0.5 * nt * a ** 2 * np.sin(dth)      # inscribed polygon, quarter
    return J, pb.area.sum(), area_hole


def estimate(a, nt, sigma_inf, R=R_OUT):
    """Numerical D_T J = [J(Omega_a) - J(Omega_0)] / |hole|."""
    D = fem.hooke_2d(E0, NU, MODEL)
    eps_inf = np.linalg.solve(D, sigma_inf)
    energy_density = float(sigma_inf @ eps_inf)       # sigma:eps with Voigt pairing

    J_a, area_ring, area_hole = _quarter_annulus_compliance(a, R, nt, sigma_inf)
    J_0 = energy_density * (area_ring + area_hole)    # exact for the solid domain
    return (J_a - J_0) / area_hole


def compliance(a, nt, sigma_inf, R=R_OUT):
    """Raw FEM compliance of the holed quarter domain (for mesh convergence)."""
    return _quarter_annulus_compliance(a, R, nt, sigma_inf)[0]


def check_equivalent_forms(rng=None, n=2000):
    """The three algebraic forms of D_T J in the post must agree identically.

        stress form     k [ 4 sigma:sigma - (tr sigma)^2 ]
        principal form  k [ 3(s_I^2 + s_II^2) - 2 s_I s_II ]
        polarization    (1/(1+nu)) [ 4 sigma:eps - (1-3nu)/(1-nu) tr sigma tr eps ]
                        (plane stress; the plane-strain factor is (1-4nu)/(1-2nu))
    """
    rng = rng or np.random.default_rng(0)
    worst = 0.0
    for model in ("plane_stress", "plane_strain"):
        for nu in (0.0, 0.2, 1.0 / 3.0, 0.45):
            D = fem.hooke_2d(E0, nu, model)
            sig = rng.normal(size=(n, 3))
            eps = sig @ np.linalg.inv(D).T
            a = fem.topological_derivative(sig, E0, nu, model)

            # principal stresses of the 2x2 tensor
            m, r = 0.5 * (sig[:, 0] + sig[:, 1]), np.hypot(
                0.5 * (sig[:, 0] - sig[:, 1]), sig[:, 2])
            sI, sII = m + r, m - r
            k = 1.0 / E0 if model == "plane_stress" else (1.0 - nu ** 2) / E0
            b = k * (3.0 * (sI ** 2 + sII ** 2) - 2.0 * sI * sII)

            se = np.einsum("ij,ij->i", sig, eps)          # sigma:eps (Voigt pairing)
            trs, tre = sig[:, 0] + sig[:, 1], eps[:, 0] + eps[:, 1]
            if model == "plane_stress":
                c = (4.0 * se - (1 - 3 * nu) / (1 - nu) * trs * tre) / (1 + nu)
            else:
                c = 4 * (1 - nu) * se \
                    - (1 - nu) * (1 - 4 * nu) / (1 - 2 * nu) * trs * tre
            scale = np.abs(a).max()
            worst = max(worst, np.abs(a - b).max() / scale,
                        np.abs(a - c).max() / scale)
    print(f"C. algebraic forms agree to {worst:.2e} (relative, 8 material settings)")
    return worst


def main():
    import matplotlib.pyplot as plt
    style.use()

    cases = {r"uniaxial  $(1,0)$": np.array([1.0, 0.0, 0.0]),
             r"biaxial   $(1,1)$": np.array([1.0, 1.0, 0.0]),
             r"shear    $(1,-1)$": np.array([1.0, -1.0, 0.0])}
    radii = [0.20, 0.10, 0.05, 0.025]
    nt_ref = 120

    print(f"Plane stress, E = {E0}, nu = {NU}, R = {R_OUT}")
    print("\nA. geometric limit a/R -> 0   (n_theta = %d)" % nt_ref)
    print("   ratio = numerical D_T J / formula;  the residual is the")
    print("   O(a^2/R^2) effect of a finite outer radius, not a formula error.\n")
    print("   %-20s %9s" % ("remote sigma", "formula") +
          "".join("%11s" % f"a/R={a:g}" for a in radii))
    ratios = {}
    for name, s in cases.items():
        pred = float(fem.topological_derivative(s, E0, NU, MODEL)[0])
        ratios[name] = [estimate(a, nt_ref, s) / pred for a in radii]
        print("   %-20s %9.5f" % (name, pred) +
              "".join("%11.5f" % v for v in ratios[name]))
    print("   %-20s %9s" % ("exact, biaxial", "-") +
          "".join("%11.5f" % (1.0 / (1.0 - (a / R_OUT) ** 2)) for a in radii))
    print("   (last row: the closed-form Lame value 1/(1 - a^2/R^2) for the"
          " biaxial case)")

    print("\n   plane strain spot check at a/R = 0.05:")
    for name, s in cases.items():
        pred = float(fem.topological_derivative(s, E0, NU, "plane_strain")[0])
        num = _plane_strain_estimate(0.05, nt_ref, s)
        print("   %-20s formula %9.5f   numerical %9.5f   ratio %8.5f"
              % (name, pred, num, num / pred))

    print("\nB. mesh limit h -> 0 of J(Omega_a) at a/R = 0.1")
    nts = [30, 60, 120, 240]
    print("   %-20s" % "remote sigma" +
          "".join("%12s" % f"nt={n}" for n in nts) + "%14s" % "Richardson")
    mesh_err = {}
    for name, s in cases.items():
        Js = [compliance(0.1, n, s) for n in nts]
        J_inf = Js[-1] + (Js[-1] - Js[-2]) / 3.0        # 2nd-order extrapolation
        mesh_err[name] = [abs(J - J_inf) / abs(J_inf) for J in Js]
        print("   %-20s" % name + "".join("%12.3e" % v for v in mesh_err[name]) +
              "%14.6f" % J_inf)
    print("   (entries: |J_h - J_inf| / |J_inf|;  at nt = 120 the")
    print("    discretization error is ~1e-4, well under the geometric error"
          " in table A)")

    print()
    check_equivalent_forms()

    colors = [style.BLUE, style.ORANGE, style.RED]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    ax = axes[0]
    for (name, row), c in zip(ratios.items(), colors):
        ax.plot(radii, np.abs(np.array(row) - 1.0), "o-", color=c, label=name,
                markeredgecolor=style.SURFACE, markeredgewidth=1.0)
    ax.set_xlabel(r"hole radius  $a/R$")
    ax.set_ylabel(r"relative error of $D_TJ$")
    ax.set_title(r"Geometric limit  $a/R\to 0$   ($n_\theta=120$)")
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes[1]
    dth = [0.5 * np.pi / n for n in nts]
    for (name, row), c in zip(mesh_err.items(), colors):
        ax.plot(dth[:-1], row[:-1], "o-", color=c, label=name,
                markeredgecolor=style.SURFACE, markeredgewidth=1.0)
    ax.set_xlabel(r"mesh size  $\Delta\theta$")
    ax.set_ylabel(r"relative error of $J(\Omega_a)$")
    ax.set_title(r"Mesh limit  $h\to 0$   ($a/R=0.1$)")

    for ax, xs, series in [(axes[0], radii, ratios), (axes[1], dth[:-1], mesh_err)]:
        lo = min(min(abs(np.asarray(r[:len(xs)]) - (1.0 if series is ratios else 0.0)))
                 for r in series.values())
        x = np.asarray(xs, float)
        y0 = 0.30 * lo * (x.max() / x.min()) ** 2
        ax.plot(x, y0 * (x / x.max()) ** 2, "--", lw=1.2, color=style.MUTED, zorder=0)
        ax.annotate("slope 2", (x.min(), y0 * (x.min() / x.max()) ** 2),
                    color=style.MUTED, fontsize=8.5,
                    textcoords="offset points", xytext=(6, -3))
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.tight_layout()
    out = "../../content/posts/topological_derivative/td_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


def _plane_strain_estimate(a, nt, sigma_inf, R=R_OUT):
    """`estimate` for the plane-strain model (used only for the spot check)."""
    global MODEL
    saved, MODEL = MODEL, "plane_strain"
    try:
        return estimate(a, nt, sigma_inf, R)
    finally:
        MODEL = saved


if __name__ == "__main__":
    main()
