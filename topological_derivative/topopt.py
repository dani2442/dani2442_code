"""
Compliance minimization driven by the topological derivative.

The design is a material indicator  chi : cells -> {0, 1}  on the triangulated
design domain.  At every iteration we

  1. solve the elasticity problem on the current design (void cells carry the
     ersatz modulus E_min = 1e-6 E, a numerical stand-in for a traction-free
     hole);
  2. evaluate the topological derivative of the compliance at every *solid*
     point, D_T J = (1/E)[4 sigma:sigma - (tr sigma)^2] - the exact cost, per
     unit area, of nucleating an infinitesimal traction-free circular hole
     there;
  3. regularize it with a cone filter of radius r_min (this both fixes the
     minimum feature size and extends the field into the void, which is what
     lets material be re-inserted);
  4. keep the cells with the largest filtered gradient, up to the volume
     allowed by a shrinking schedule V_k.

Step 4 is the Lagrangian form of the volume constraint: for L = J + l |Omega|
the change under hole nucleation is |B|(D_T J - l), so cells with
D_T J < l should be voided, and the multiplier l is exactly the V_k-quantile of
the filtered gradient.  Steps 3-4 with the sensitivity averaged over two
iterations are the BESO update of Huang & Xie, with the topological derivative
in place of their heuristic sensitivity.

Outputs (written to ../../content/posts/topological_derivative/):
    td_mesh.png        the triangulation and boundary conditions
    td_gradient.png    the topological gradient on the full domain
    td_steps.png       five steps of the optimization + convergence history
    td_examples.png    final designs for two further load cases
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

import fem
import style

E_SOLID, E_MIN, NU = 1.0, 1e-6, 0.3
MODEL = "plane_stress"
OUT = "../../content/posts/topological_derivative/"


# -----------------------------------------------------------------------------
# Problem definitions
# -----------------------------------------------------------------------------
@dataclass
class Problem:
    name: str
    lx: float
    ly: float
    nx: int
    ny: int
    vol_frac: float
    rmin_cells: float = 3.5
    label: str = ""
    # filled in by `build`
    mesh: tuple = field(default=None, repr=False)
    pb: fem.Elasticity2D = field(default=None, repr=False)
    f: np.ndarray = field(default=None, repr=False)
    fixed: np.ndarray = field(default=None, repr=False)
    void: np.ndarray = field(default=None, repr=False)
    keep: np.ndarray = field(default=None, repr=False)
    anchors: list = field(default_factory=list, repr=False)

    # -- geometry helpers -----------------------------------------------------
    @property
    def h(self):
        return np.array([self.lx / self.nx, self.ly / self.ny])

    def cell_centers(self):
        hx, hy = self.h
        xc = (np.arange(self.nx) + 0.5) * hx
        yc = (np.arange(self.ny) + 0.5) * hy
        return np.meshgrid(xc, yc, indexing="ij")

    def node_grid(self):
        return np.arange((self.nx + 1) * (self.ny + 1)).reshape(self.nx + 1, self.ny + 1)


def _edges_on_line(line, mask):
    """Node-pair edges along a boundary grid line, kept where `mask` holds."""
    pairs = np.column_stack([line[:-1], line[1:]])
    return pairs[mask[:-1] & mask[1:]]


def cantilever(nx=150, ny=75):
    p = Problem("cantilever", 2.0, 1.0, nx, ny, 0.40,
                label="clamped left edge, tip load")
    p.mesh = fem.rect_mesh(p.lx, p.ly, nx, ny)
    nodes, tris, cell, _ = p.mesh
    p.pb = fem.Elasticity2D(nodes, tris, nu=NU, model=MODEL)
    nid = p.node_grid()

    # Clamp the whole left edge.
    p.fixed = np.concatenate([2 * nid[0], 2 * nid[0] + 1])

    # Downward traction of total magnitude 1 on a patch at the right edge.
    y = nodes[nid[nx], 1]
    half = max(0.06 * p.ly, 1.01 * p.h[1])      # at least one element edge
    patch = np.abs(y - 0.5 * p.ly) <= half
    edges = _edges_on_line(nid[nx], patch)
    length = np.abs(y[patch].max() - y[patch].min())
    p.f = p.pb.edge_load(edges, np.array([0.0, -1.0 / length]))

    Xc, Yc = p.cell_centers()
    p.void = np.zeros(nx * ny, bool)
    p.keep = ((Xc > p.lx - 3 * p.h[0]) &
              (np.abs(Yc - 0.5 * p.ly) < half + 2 * p.h[1])).ravel()
    p.anchors = [("load", (p.lx, 0.5 * p.ly))]
    return p


def mbb_beam(nx=180, ny=60):
    p = Problem("mbb", 3.0, 1.0, nx, ny, 0.40,
                label="simply supported, central top load")
    p.mesh = fem.rect_mesh(p.lx, p.ly, nx, ny)
    nodes, tris, cell, _ = p.mesh
    p.pb = fem.Elasticity2D(nodes, tris, nu=NU, model=MODEL)
    nid = p.node_grid()

    pin = nid[0, 0]                       # bottom-left: both components
    roller = nid[nx, 0]                   # bottom-right: vertical only
    p.fixed = np.array([2 * pin, 2 * pin + 1, 2 * roller + 1])

    x = nodes[nid[:, ny], 0]
    half = max(0.05 * p.ly, 1.01 * p.h[0])      # at least one element edge
    patch = np.abs(x - 0.5 * p.lx) <= half
    edges = _edges_on_line(nid[:, ny], patch)
    length = np.abs(x[patch].max() - x[patch].min())
    p.f = p.pb.edge_load(edges, np.array([0.0, -1.0 / length]))

    Xc, Yc = p.cell_centers()
    hx, hy = p.h
    p.void = np.zeros(nx * ny, bool)
    p.keep = (((Yc > p.ly - 3 * hy) & (np.abs(Xc - 0.5 * p.lx) < half + 2 * hx)) |
              ((Yc < 3 * hy) & ((Xc < 3 * hx) | (Xc > p.lx - 3 * hx)))).ravel()
    p.anchors = [("load", (0.5 * p.lx, p.ly)), ("support", (0.0, 0.0)),
                 ("support", (p.lx, 0.0))]
    return p


def l_bracket(n=120):
    p = Problem("lbracket", 1.0, 1.0, n, n, 0.40, rmin_cells=3.5,
                label="L-shaped domain, re-entrant corner")
    p.mesh = fem.rect_mesh(p.lx, p.ly, n, n)
    nodes, tris, cell, _ = p.mesh
    p.pb = fem.Elasticity2D(nodes, tris, nu=NU, model=MODEL)
    nid = p.node_grid()
    cut = 0.4

    # Permanently void upper-right block -> an L-shaped design domain.
    Xc, Yc = p.cell_centers()
    p.void = ((Xc > cut) & (Yc > cut)).ravel()

    # Clamp the part of the top edge that belongs to the vertical arm.
    top = nid[:, n][nodes[nid[:, n], 0] <= cut + 1e-12]
    p.fixed = np.concatenate([2 * top, 2 * top + 1])

    y = nodes[nid[n], 1]
    half = max(0.05, 1.01 * p.h[1])             # at least one element edge
    patch = (y <= cut + 1e-12) & (y >= cut - 2 * half)
    edges = _edges_on_line(nid[n], patch)
    length = np.abs(y[patch].max() - y[patch].min())
    p.f = p.pb.edge_load(edges, np.array([0.0, -1.0 / length]))

    hx, hy = p.h
    p.keep = ((Xc > p.lx - 3 * hx) & (Yc <= cut) & (Yc > cut - 2 * half - 2 * hy)).ravel()
    p.anchors = [("load", (p.lx, cut))]
    return p


# -----------------------------------------------------------------------------
# Cone filter on the cell grid
# -----------------------------------------------------------------------------
def cone_filter(nx, ny, hx, hy, rmin):
    """Row-normalized linear-hat (cone) averaging operator of radius rmin."""
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []
    for di in range(-int(rmin / hx), int(rmin / hx) + 1):
        for dj in range(-int(rmin / hy), int(rmin / hy) + 1):
            w = rmin - np.hypot(di * hx, dj * hy)
            if w <= 0.0:
                continue
            i0, i1 = max(0, -di), min(nx, nx - di)
            j0, j1 = max(0, -dj), min(ny, ny - dj)
            if i0 >= i1 or j0 >= j1:
                continue
            src = idx[i0:i1, j0:j1].ravel()
            dst = idx[i0 + di:i1 + di, j0 + dj:j1 + dj].ravel()
            rows.append(dst)
            cols.append(src)
            vals.append(np.full(src.size, w))
    H = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nx * ny, nx * ny)).tocsr()
    return sp.diags(1.0 / np.asarray(H.sum(axis=1)).ravel()) @ H


# -----------------------------------------------------------------------------
# The optimization loop
# -----------------------------------------------------------------------------
def solve_state(p, chi):
    """Solve on the design `chi` and return (u, J, per-triangle D_T J)."""
    E_cell = np.where(chi > 0.5, E_SOLID, E_MIN)
    E_cell[p.void] = E_MIN
    _, _, cell, _ = p.mesh
    u, J = p.pb.solve(E_cell[cell], p.f, p.fixed)
    # The formula is exact at points of the solid phase, where the recovered
    # stress is the physical one; evaluate it with the solid moduli everywhere
    # and mask the void afterwards.
    sigma = p.pb.stress(u, E_SOLID)
    return u, J, fem.topological_derivative(sigma, E_SOLID, NU, MODEL)


def optimize(p, n_iter=90, evol_rate=0.02, snapshots=(0.90, 0.75, 0.60, 0.50),
             verbose=True):
    _, _, cell, _ = p.mesh
    nc = p.nx * p.ny
    hx, hy = p.h
    H = cone_filter(p.nx, p.ny, hx, hy, p.rmin_cells * max(hx, hy))

    design_area = float((~p.void).sum())
    chi = (~p.void).astype(float)
    g_prev = None
    hist = {"J": [], "V": []}
    frames, wanted = [], list(snapshots)

    # Cell areas are all equal on a uniform grid, so volume fractions are counts.
    for k in range(n_iter):
        u, J, g_tri = solve_state(p, chi)

        # Triangle -> cell (two triangles of equal area per cell).
        g_cell = np.zeros(nc)
        np.add.at(g_cell, cell, 0.5 * g_tri)
        g_cell *= chi                       # rigorous only on the solid phase
        g = H @ g_cell
        g = g if g_prev is None else 0.5 * (g + g_prev)
        g_prev = g

        V = chi.sum() / design_area
        hist["J"].append(J)
        hist["V"].append(V)
        if verbose and (k % 10 == 0 or k == n_iter - 1):
            print(f"   it {k:3d}   V = {V:5.3f}   J = {J:.6g}")

        while wanted and V <= wanted[0] + 1e-9:
            frames.append((k, V, J, chi.copy()))
            wanted.pop(0)

        # Shrink the volume, then hold it while the topology settles.
        V_next = max(p.vol_frac, V * (1.0 - evol_rate))
        n_keep = int(round(V_next * design_area))

        chi_new = np.zeros(nc)
        chi_new[p.keep & ~p.void] = 1.0
        n_forced = int(chi_new.sum())
        free = np.where(~p.void & ~p.keep)[0]
        order = free[np.argsort(-g[free])]
        chi_new[order[:max(0, n_keep - n_forced)]] = 1.0
        chi = chi_new

    u, J, _ = solve_state(p, chi)
    hist["J"].append(J)
    hist["V"].append(chi.sum() / design_area)
    frames.append((n_iter, hist["V"][-1], J, chi.copy()))
    if verbose:
        print(f"   final  V = {hist['V'][-1]:5.3f}   J = {J:.6g}")
    return chi, frames, hist


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def _tri(p):
    import matplotlib.tri as mtri
    nodes, tris, _, _ = p.mesh
    return mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)


def _bare(ax, p):
    ax.set_aspect("equal")
    ax.set_xlim(-0.05 * p.lx, 1.05 * p.lx)
    ax.set_ylim(-0.07 * p.ly, 1.07 * p.ly)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def outline_domain(ax, p, **kw):
    """Thin outline of the design domain (the outer box plus any fixed void)."""
    hx, hy = p.h
    m = np.pad((~p.void).reshape(p.nx, p.ny).astype(float), 1)
    xs = np.concatenate([[-0.5 * hx], (np.arange(p.nx) + 0.5) * hx, [p.lx + 0.5 * hx]])
    ys = np.concatenate([[-0.5 * hy], (np.arange(p.ny) + 0.5) * hy, [p.ly + 0.5 * hy]])
    ax.contour(xs, ys, m.T, levels=[0.5],
               colors=kw.pop("color", style.MUTED), linewidths=kw.pop("lw", 0.9))


def plot_design(ax, p, chi):
    from matplotlib.colors import ListedColormap
    _, _, cell, _ = p.mesh
    val = np.where(p.void, 0.0, chi)[cell]
    ax.tripcolor(_tri(p), facecolors=val, cmap=ListedColormap([style.VOID, style.SOLID]),
                 vmin=0.0, vmax=1.0, rasterized=True)
    outline_domain(ax, p)
    _bare(ax, p)


def plot_field(ax, p, g_tri, clip=98.0):
    from matplotlib.colors import PowerNorm
    vmax = np.percentile(g_tri, clip)
    m = ax.tripcolor(_tri(p), facecolors=np.minimum(g_tri, vmax), cmap=style.SEQ,
                     norm=PowerNorm(0.45, vmin=0.0, vmax=vmax), rasterized=True)
    _bare(ax, p)
    return m


def annotate_bcs(ax, p, nodes):
    """Mark the clamped dofs and the loaded patch."""
    fixed_nodes = np.unique(np.asarray(p.fixed) // 2)
    ax.plot(nodes[fixed_nodes, 0], nodes[fixed_nodes, 1], "|", ms=7, mew=1.6,
            color=style.INK_2, zorder=5)
    loaded = np.unique(np.where(np.abs(p.f) > 0)[0] // 2)
    fx = p.f[2 * loaded]
    fy = p.f[2 * loaded + 1]
    scale = 0.22 * p.ly / max(np.hypot(fx, fy).max(), 1e-30)
    for n, gx, gy in zip(loaded, fx, fy):
        ax.annotate("", xy=nodes[n], xytext=nodes[n] - scale * np.array([gx, gy]),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color=style.ORANGE,
                                    mutation_scale=8), zorder=6)


def figure_mesh():
    """The triangulated design domain and its boundary conditions."""
    import matplotlib.pyplot as plt
    p = cantilever(24, 12)
    nodes, tris, _, _ = p.mesh
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.triplot(_tri(p), lw=0.55, color=style.MUTED, alpha=0.55)
    annotate_bcs(ax, p, nodes)
    _bare(ax, p)
    ax.text(-0.035 * p.lx, 0.5 * p.ly, r"$\Gamma_D$", ha="right", va="center",
            color=style.INK_2, fontsize=11)
    ax.text(p.lx, 0.5 * p.ly + 0.27 * p.ly, r"$\Gamma_N$", ha="center", va="bottom",
            color=style.ORANGE, fontsize=11)
    ax.set_title(r"Design domain $\Omega_0$, shown at $24\times12$ cells "
                 r"(%d nodes, %d triangles)" % (len(nodes), len(tris)) + "\n"
                 r"the runs below use $150\times75$ cells",
                 color=style.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT + "td_mesh.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote td_mesh.png")


def figure_gradient(p):
    """The topological gradient on the full-material domain."""
    import matplotlib.pyplot as plt
    _, _, g_tri = solve_state(p, np.ones(p.nx * p.ny))
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    m = plot_field(ax, p, g_tri)
    cb = fig.colorbar(m, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label(r"$D_TJ(\hat x)$", color=style.INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=style.MUTED, labelcolor=style.INK_2)
    ax.set_title(r"Topological gradient of the compliance on $\Omega_0$"
                 "\n" r"dark $=$ expensive to perforate,  light $=$ nearly free",
                 color=style.INK)
    fig.tight_layout()
    fig.savefig(OUT + "td_gradient.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote td_gradient.png")


def figure_steps(p, frames, hist):
    """Five steps of the optimization plus the convergence history."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 7.6))
    flat = axes.ravel()
    J0 = hist["J"][0]
    for ax, (k, V, J, chi) in zip(flat, frames):
        plot_design(ax, p, chi)
        ax.set_title(r"step %d:   $|\Omega|/|\Omega_0| = %.2f$,   $J/J_0 = %.2f$"
                     % (k, V, J / J0), color=style.INK, fontsize=10)

    ax = flat[5]
    ax.grid(True)
    it = np.arange(len(hist["J"]))
    ax.plot(it, hist["V"], color=style.BLUE, label=r"volume  $|\Omega_k|/|\Omega_0|$")
    ax.plot(it, J0 / np.asarray(hist["J"]), color=style.ORANGE,
            label=r"stiffness  $J_0/J(\Omega_k)$")
    ks = [f[0] for f in frames]
    ax.plot(ks, [hist["V"][k] for k in ks], "o", color=style.BLUE,
            markeredgecolor=style.SURFACE, markeredgewidth=1.0, zorder=5)
    ax.plot(ks, [J0 / hist["J"][k] for k in ks], "o", color=style.ORANGE,
            markeredgecolor=style.SURFACE, markeredgewidth=1.0, zorder=5)
    ax.set_xlabel("iteration")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Convergence (both scales dimensionless)", fontsize=10)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT + "td_steps.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote td_steps.png")


def figure_examples(results):
    """Final designs for the additional load cases."""
    import matplotlib.pyplot as plt
    ratios = [p.lx / p.ly for p, _, _ in results]
    fig, axes = plt.subplots(1, len(results), figsize=(2.1 * sum(ratios) + 1.4, 3.3),
                             gridspec_kw={"width_ratios": ratios})
    for ax, (p, chi, hist) in zip(np.atleast_1d(axes), results):
        plot_design(ax, p, chi)
        ax.set_title("%s\n" % p.label +
                     r"$|\Omega|/|\Omega_0| = %.2f$,   $J/J_0 = %.2f$"
                     % (hist["V"][-1], hist["J"][-1] / hist["J"][0]),
                     color=style.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT + "td_examples.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote td_examples.png")


def main():
    style.use()
    figure_mesh()

    print("cantilever:")
    cant = cantilever()
    figure_gradient(cant)
    chi, frames, hist = optimize(cant)
    figure_steps(cant, frames, hist)

    extra = []
    for builder in (mbb_beam, l_bracket):
        p = builder()
        print(f"{p.name}:")
        chi_p, _, hist_p = optimize(p, snapshots=())
        extra.append((p, chi_p, hist_p))
    figure_examples(extra)


if __name__ == "__main__":
    main()
