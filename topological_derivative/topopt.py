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
the filtered gradient. Steps 3-4 follow the BESO update of Huang & Xie, using
recursive temporal averaging and the topological derivative in place of their
heuristic sensitivity.

Outputs (written to ../../content/posts/topological_derivative/):
    td_mesh.png        the triangulation and boundary conditions
    td_gradient.png    the topological gradient on the full domain
    td_optimization.gif  actual designs, filtered sensitivities and history
    td_convergence.png   static convergence history
    td_bridge.png        three-pier bridge with piers and deck traction
    td_lbracket.png      L-bracket with clamp and traction
    td_bridge_sweep.png  bridge designs across volume and filter radius
    td_results.json      parameters and numerical histories behind the figures
"""

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json

import numpy as np
import scipy.sparse as sp

import fem
import style

E_SOLID, E_MIN, NU = 1.0, 1e-6, 0.3
MODEL = "plane_stress"
OUT = Path(__file__).resolve().parents[2] / "content/posts/topological_derivative"


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


def bridge(nx=180, ny=60):
    p = Problem("bridge", 3.0, 1.0, nx, ny, 0.40,
                label="three pinned piers, uniformly loaded deck")
    p.mesh = fem.rect_mesh(p.lx, p.ly, nx, ny)
    nodes, tris, cell, _ = p.mesh
    p.pb = fem.Elasticity2D(nodes, tris, nu=NU, model=MODEL)
    nid = p.node_grid()

    # Three piers on the bottom edge: the two abutments and one at midspan.
    # Each restrains horizontal and vertical translation, so both spans are
    # continuous over supports that can carry a horizontal reaction.
    pier_x = np.array([0.0, 0.5 * p.lx, p.lx])
    pins = nid[np.rint(pier_x / p.h[0]).astype(int), 0]
    p.fixed = np.concatenate([2 * pins, 2 * pins + 1])

    # Gamma_N is the whole top edge: a uniform downward deck load, scaled to
    # total magnitude 1 so J/J_0 stays comparable with the other load cases.
    top = nid[:, ny]
    edges = _edges_on_line(top, np.ones(nx + 1, bool))
    p.f = p.pb.edge_load(edges, np.array([0.0, -1.0 / p.lx]))

    Xc, Yc = p.cell_centers()
    hx, hy = p.h
    p.void = np.zeros(nx * ny, bool)
    # The deck carries the traction and the pier heads receive the reactions;
    # both must stay solid or the greedy step can load pure ersatz material.
    deck = Yc > p.ly - 2 * hy
    piers = (Yc < 3 * hy) & (np.abs(Xc[..., None] - pier_x).min(-1) < 3 * hx)
    p.keep = (deck | piers).ravel()
    p.anchors = ([("load", (0.5 * p.lx, p.ly))] +
                 [("support", (x, 0.0)) for x in pier_x])
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

    # Gamma_N is horizontal: the right end of the upper face of the horizontal
    # arm, at y = cut, carrying a downward traction normal to that face.
    line = nid[:, int(round(cut / p.h[1]))]
    x = nodes[line, 0]
    half = max(0.05, 1.01 * p.h[0])             # at least one element edge
    patch = x >= p.lx - 2 * half - 1e-12
    edges = _edges_on_line(line, patch)
    length = np.abs(x[patch].max() - x[patch].min())
    p.f = p.pb.edge_load(edges, np.array([0.0, -1.0 / length]))

    hx, hy = p.h
    p.keep = ((Yc <= cut) & (Yc > cut - 3 * hy) &
              (Xc > p.lx - 2 * half - 2 * hx)).ravel()
    p.anchors = [("load", (p.lx - half, cut))]
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
             verbose=True, record=False):
    """Return final design, selected snapshots and history of solved states.

    With record=True, also retain each design and the filtered, temporally
    averaged score used for its next update. The final score is evaluated on
    the final design, so GIF frames never pair a design with a stale field.
    """
    _, _, cell, _ = p.mesh
    nc = p.nx * p.ny
    hx, hy = p.h
    H = cone_filter(p.nx, p.ny, hx, hy, p.rmin_cells * max(hx, hy))

    design_area = float((~p.void).sum())
    chi = (~p.void).astype(float)
    g_prev = None
    hist = {"J": [], "V": []}
    if record:
        hist.update(chi=[], G=[])
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
        if record:
            hist["chi"].append(chi.copy())
            hist["G"].append(g.copy())
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

    u, J, g_tri = solve_state(p, chi)
    hist["J"].append(J)
    hist["V"].append(chi.sum() / design_area)
    if record:
        g_cell = np.zeros(nc)
        np.add.at(g_cell, cell, 0.5 * g_tri)
        g = H @ (g_cell * chi)
        hist["chi"].append(chi.copy())
        hist["G"].append(g if g_prev is None else 0.5 * (g + g_prev))
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
    ax.set_xlim(-0.13 * p.ly, p.lx + 0.23 * p.ly)
    ax.set_ylim(-0.32 * p.ly if p.name == "bridge" else -0.05 * p.ly,
                1.36 * p.ly if p.name == "bridge" else 1.16 * p.ly)
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
    """Draw displacement constraints separately from the prescribed traction.

    The bridge has three pinned point supports, not clamped edges. Its point
    constraints belong to the discrete benchmark; Gamma_D labels constrained
    components.
    Arrows illustrate traction direction, not the unequal nodal quadrature
    weights at the endpoints of a uniformly loaded edge.
    """
    from matplotlib.patches import Polygon
    fixed_nodes = np.unique(np.asarray(p.fixed) // 2)
    if p.name == "bridge":
        for x in np.unique(nodes[fixed_nodes, 0]):
            ax.add_patch(Polygon([(x, 0), (x - .065, -.11), (x + .065, -.11)],
                                 facecolor=style.SURFACE, edgecolor=style.INK_2,
                                 lw=1.3, zorder=6))
            ground = -.13
            ax.plot([x - .09, x + .09], [ground, ground], color=style.INK_2, lw=1)
        ax.text(.5 * p.lx, -.155, r"$\Gamma_D:\ u_x=u_y=0$ at three piers",
                ha="center", va="top", fontsize=8, color=style.INK_2)
    else:
        xy = nodes[fixed_nodes]
        ax.plot(xy[:, 0], xy[:, 1], color=style.INK_2, lw=2.5, zorder=5)
        sample = xy[np.linspace(0, len(xy) - 1, min(13, len(xy)), dtype=int)]
        for x, y in sample:
            dx, dy = (-.025, -.025) if p.name == "cantilever" else (.025, .025)
            ax.plot([x, x + dx], [y, y + dy], color=style.INK_2, lw=1, zorder=5)
        if p.name == "cantilever":
            ax.text(-.07, .5 * p.ly, r"$\Gamma_D$", ha="right", va="center",
                    color=style.INK_2, fontsize=10)
        else:
            ax.text(.2, 1.07, r"$\Gamma_D:\ u=0$", ha="center",
                    color=style.INK_2, fontsize=10)
    loaded = np.unique(np.where(np.abs(p.f) > 0)[0] // 2)
    xy = nodes[loaded]
    ax.plot(xy[:, 0], xy[:, 1], color=style.ORANGE, lw=3, zorder=6)
    if p.name == "bridge":
        for x in np.linspace(xy[:, 0].min(), xy[:, 0].max(), 13):
            ax.annotate("", xy=(x, p.ly), xytext=(x, p.ly + .22),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2,
                                        color=style.ORANGE), zorder=7)
        ax.text(p.lx / 2, p.ly + .27,
                r"$\Gamma_N$: uniform $g\downarrow$ on the whole top edge",
                ha="center", color=style.ORANGE, fontsize=9.5)
    elif p.name == "lbracket":
        # The loaded patch is horizontal, so the arrows press down onto it
        # from the permanently void block above.
        y0 = xy[:, 1].mean()
        for x in np.linspace(xy[:, 0].min(), xy[:, 0].max(), 6):
            ax.annotate("", xy=(x, y0), xytext=(x, y0 + .17),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2,
                                        color=style.ORANGE), zorder=7)
        ax.text(xy[:, 0].mean(), y0 + .21, r"$\Gamma_N$", ha="center",
                color=style.ORANGE, fontsize=10)
    else:
        center = xy[:, 1].mean()
        # A leader identifies the loaded vertical patch; the adjacent arrow
        # displays the (tangential) downward traction without hiding the patch.
        ax.plot([p.lx, p.lx + .10], [center, center], lw=.9, color=style.ORANGE)
        ax.annotate("", xy=(p.lx + .10, center - .13),
                    xytext=(p.lx + .10, center + .13),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5,
                                    color=style.ORANGE), zorder=7)
        ax.text(p.lx + .10, center + .16, r"$\Gamma_N$", ha="center",
                color=style.ORANGE, fontsize=10)


def figure_mesh():
    """The triangulated design domain and its boundary conditions."""
    import matplotlib.pyplot as plt
    p = cantilever(24, 12)
    nodes, tris, _, _ = p.mesh
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.triplot(_tri(p), lw=0.55, color=style.MUTED, alpha=0.55)
    annotate_bcs(ax, p, nodes)
    _bare(ax, p)
    ax.set_title(r"Design domain $\Omega_0$, shown at $24\times12$ cells "
                 r"(%d nodes, %d triangles)" % (len(nodes), len(tris)) + "\n"
                 r"the runs below use $150\times75$ cells",
                 color=style.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "td_mesh.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote td_mesh.png")


def figure_gradient(p):
    """The topological gradient on the full-material domain."""
    import matplotlib.pyplot as plt
    _, _, g_tri = solve_state(p, np.ones(p.nx * p.ny))
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    m = plot_field(ax, p, g_tri)
    annotate_bcs(ax, p, p.mesh[0])
    cb = fig.colorbar(m, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label(r"$D_TJ(\hat x)$", color=style.INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=style.MUTED, labelcolor=style.INK_2)
    ax.set_title(r"Topological gradient of the compliance on $\Omega_0$"
                 "\n" r"dark $=$ expensive to perforate,  light $=$ nearly free",
                 color=style.INK)
    fig.tight_layout()
    fig.savefig(OUT / "td_gradient.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote td_gradient.png")


def plot_history(ax, hist, stop=None):
    """Plot only states already reached, including the initial full domain."""
    stop = len(hist["J"]) if stop is None else stop
    it = np.arange(stop)
    ax.plot(it, hist["V"][:stop], color=style.BLUE,
            label=r"material fraction $|\Omega_k|/|\Omega_0|$")
    ax.plot(it, hist["J"][0] / np.asarray(hist["J"][:stop]), color=style.ORANGE,
            label=r"relative stiffness $J_0/J_k$")
    ax.set_xlim(0, len(hist["J"]) - 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("iteration")
    ax.legend(loc="lower left", fontsize=8)


def figure_convergence(hist):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    plot_history(ax, hist)
    ax.set_title("Cantilever: material use and stiffness")
    fig.tight_layout()
    fig.savefig(OUT / "td_convergence.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def figure_animation(p, hist):
    """Render every solved iteration, with a fixed colour scale across frames."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from PIL import Image

    fig = plt.figure(figsize=(9.6, 5.6), dpi=110)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.65, 1], hspace=.35, wspace=.15)
    design_ax, field_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    history_ax = fig.add_subplot(grid[1, :])
    fig.subplots_adjust(left=.06, right=.96, bottom=.09, top=.85)
    title = fig.suptitle("", fontsize=12, y=.98)
    scores = np.asarray(hist["G"])
    positive = scores[scores > 0]
    vmax = np.percentile(positive, 98)
    norm = PowerNorm(.45, vmin=0, vmax=vmax, clip=True)
    # A shared palette keeps the background and colour ramp stable in the GIF.
    from matplotlib.colors import to_rgb
    colors = [to_rgb(c) for c in (style.SURFACE, style.INK, style.INK_2,
                                  style.ORANGE, style.BLUE, style.SOLID, style.GRID)]
    colors += [tuple(c[:3]) for c in style.SEQ(np.linspace(0, 1, 160))]
    colors += [(x, x, x) for x in np.linspace(0, 1, 64)]
    colors += [to_rgb(style.SURFACE)] * (256 - len(colors))
    palette = Image.new("P", (1, 1))
    palette.putpalette(np.rint(255 * np.asarray(colors)).astype("uint8").ravel().tolist())
    images = []
    for k, (chi, G, J, V) in enumerate(zip(hist["chi"], scores, hist["J"], hist["V"])):
        design_ax.clear()
        field_ax.clear()
        history_ax.clear()
        plot_design(design_ax, p, chi)
        # Cell values correspond to the two triangles sharing each quad.
        m = field_ax.imshow(G.reshape(p.nx, p.ny).T, origin="lower",
                            extent=(0, p.lx, 0, p.ly), cmap=style.SEQ, norm=norm,
                            interpolation="nearest")
        outline_domain(field_ax, p)
        _bare(field_ax, p)
        for ax in (design_ax, field_ax):
            annotate_bcs(ax, p, p.mesh[0])
        design_ax.set_title("Material on the fixed triangular mesh", fontsize=10)
        field_ax.set_title(r"Filtered update score $G_k$", fontsize=10)
        plot_history(history_ax, hist, k + 1)
        history_ax.axvline(k, color=style.MUTED, lw=.8, ls=":")
        title.set_text(f"Cantilever  |  iteration {k:02d}/{len(scores) - 1}\n"
                       f"material = {V:.0%}    |    stiffness = {hist['J'][0] / J:.1%}")
        if k == 0:
            cax = fig.add_axes([.59, .888, .29, .012])
            cb = fig.colorbar(m, cax=cax, orientation="horizontal")
            cb.set_ticks([0, vmax], labels=["low", "high"])
            cb.ax.tick_params(length=0, labelsize=8)
            cb.outline.set_visible(False)
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        images.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
    durations = [130] * len(images)
    durations[0], durations[-1] = 900, 2400
    images[0].save(OUT / "td_optimization.gif", save_all=True,
                   append_images=images[1:], duration=durations, loop=0,
                   optimize=False, disposal=2)
    plt.close(fig)
    print(f"wrote td_optimization.gif ({len(images)} solved states)")


def figure_example(p, chi, hist):
    """Each load case gets its own figure with its actual constraints."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.2) if p.name == "bridge" else (5.2, 5.2))
    plot_design(ax, p, chi)
    annotate_bcs(ax, p, p.mesh[0])
    name = ("Three-pier bridge: uniformly loaded deck over two spans"
            if p.name == "bridge" else
            "L-bracket: top clamp, downward traction on the horizontal arm")
    ax.set_title(name + "\n" +
                 r"$V=%.2f$,  $r_{\min}/h=%.1f$,  $J/J_0=%.2f$  (%d iterations)"
                 % (hist["V"][-1], p.rmin_cells,
                    hist["J"][-1] / hist["J"][0], len(hist["J"]) - 1), fontsize=10)
    fig.tight_layout()
    filename = "td_bridge.png" if p.name == "bridge" else "td_lbracket.png"
    fig.savefig(OUT / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {filename}")


def figure_bridge_sweep(results):
    """Rows change the length scale, columns change the material budget."""
    import matplotlib.pyplot as plt
    radii = sorted({p.rmin_cells for p, _, _ in results})
    volumes = sorted({p.vol_frac for p, _, _ in results})
    fig, axes = plt.subplots(len(radii), len(volumes), figsize=(13.6, 8.5),
                             squeeze=False)
    for p, chi, hist in results:
        row, col = radii.index(p.rmin_cells), volumes.index(p.vol_frac)
        ax = axes[row, col]
        plot_design(ax, p, chi)
        annotate_bcs(ax, p, p.mesh[0])
        ax.set_title(r"$V=%.2f$   $r_{\min}/h=%.1f$" % (p.vol_frac, p.rmin_cells)
                     + "\n" + r"$J/J_0=%.2f$   stiffness $=%.0f\%%$"
                     % (hist["J"][-1] / hist["J"][0],
                        100 * hist["J"][0] / hist["J"][-1]), fontsize=10)
    fig.suptitle("One bridge load case, nine final configurations\n"
                 r"same $180\times60$ cell mesh, three piers, uniform deck load,"
                 r" material and 120 iterations; evolution rate $2\%$",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=3.2, w_pad=2)
    fig.savefig(OUT / "td_bridge_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote td_bridge_sweep.png")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--cache-dir", type=Path,
                        help="reuse saved numerical runs when adjusting figure layouts")
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    style.use()
    figure_mesh()
    summary = []

    def run(p, n_iter=90, record=False):
        bc_tag = "_3pier_deck" if p.name == "bridge" else ""
        key = f"{p.name}{bc_tag}_{p.nx}x{p.ny}_v{p.vol_frac}_r{p.rmin_cells}_n{n_iter}_record{record}"
        cache = args.cache_dir / (key + ".npz") if args.cache_dir else None
        if cache and cache.exists():
            with np.load(cache) as data:
                hist = {k: data[k] for k in ("J", "V", "chi", "G") if k in data}
                chi = data["final"]
            print(f"loaded {key}")
        else:
            print(f"{key}:")
            chi, _, hist = optimize(p, n_iter=n_iter, record=record, snapshots=())
            if cache:
                np.savez_compressed(cache, final=chi, **hist)
        J = np.asarray(hist["J"])
        V = np.asarray(hist["V"])
        assert np.isfinite(J).all() and (J > 0).all()
        assert abs(V[-1] - p.vol_frac) <= 1 / (~p.void).sum()
        assert np.all(chi[p.keep & ~p.void] == 1) and np.all(chi[p.void] == 0)
        summary.append(dict(name=p.name, nx=p.nx, ny=p.ny, lx=p.lx, ly=p.ly,
                            target_volume=p.vol_frac, rmin_cells=p.rmin_cells,
                            iterations=n_iter, evolution_rate=.02, E=E_SOLID,
                            E_min=E_MIN, nu=NU, model=MODEL,
                            support_condition=("three pins: ux=uy=0 at x=0, L/2, L"
                                               if p.name == "bridge" else p.label),
                            total_force=p.f.reshape(-1, 2).sum(axis=0).tolist(),
                            J=J.tolist(), V=V.tolist()))
        return chi, hist

    cant = cantilever()
    figure_gradient(cant)
    _, hist = run(cant, record=True)
    figure_animation(cant, hist)
    figure_convergence(hist)

    bridge_results = []
    for radius in (2.5, 3.5, 5.5):
        for volume in (.30, .40, .50):
            p = bridge()
            p.rmin_cells, p.vol_frac = radius, volume
            chi, hist = run(p, n_iter=120)
            bridge_results.append((p, chi, hist))
            if radius == 3.5 and volume == .40:
                figure_example(p, chi, hist)
    figure_bridge_sweep(bridge_results)

    bracket = l_bracket()
    chi, hist = run(bracket)
    figure_example(bracket, chi, hist)
    (OUT / "td_results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
