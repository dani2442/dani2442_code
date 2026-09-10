"""
Drawing primitives.

Two groups: the design domain itself (the material indicator, magnitude fields
on the triangulation, the convergence history), and the boundary-condition
symbols that each load case in `examples/` composes into its own sketch of
Gamma_D and Gamma_N.  `figures.py` builds the post's figures out of both.
"""

import numpy as np
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap, PowerNorm
from matplotlib.patches import Polygon

import style


# -----------------------------------------------------------------------------
# The design domain
# -----------------------------------------------------------------------------
def triangulation(p):
    return mtri.Triangulation(p.nodes[:, 0], p.nodes[:, 1], p.tris)


def bare(ax, p):
    """Strip an axes down to the domain: equal aspect, no ticks, no frame."""
    ax.set_aspect("equal")
    ax.set_xlim(-0.13 * p.ly, p.lx + 0.23 * p.ly)
    lo, hi = p.ylim
    ax.set_ylim(lo * p.ly, hi * p.ly)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def outline_domain(ax, p, color=style.MUTED, lw=0.9):
    """Thin outline of the design domain (the outer box plus any fixed void)."""
    hx, hy = p.h
    m = np.pad((~p.void).reshape(p.nx, p.ny).astype(float), 1)
    xs = np.concatenate([[-0.5 * hx], (np.arange(p.nx) + 0.5) * hx,
                         [p.lx + 0.5 * hx]])
    ys = np.concatenate([[-0.5 * hy], (np.arange(p.ny) + 0.5) * hy,
                         [p.ly + 0.5 * hy]])
    ax.contour(xs, ys, m.T, levels=[0.5], colors=color, linewidths=lw)


def plot_design(ax, p, chi):
    """The material indicator on the fixed triangular mesh."""
    val = np.where(p.void, 0.0, chi)[p.cell]
    ax.tripcolor(triangulation(p), facecolors=val,
                 cmap=ListedColormap([style.VOID, style.SOLID]),
                 vmin=0.0, vmax=1.0, rasterized=True)
    outline_domain(ax, p)
    bare(ax, p)


def solid_field(p, chi, nodal):
    """A nodal field averaged onto the triangles, masked outside the material.

    What moves is the structure, so the void - where the ersatz modulus still
    admits a displacement - is left out of both the picture and the colour
    scale it is built from.
    """
    solid = np.where(p.void, 0.0, chi)[p.cell] >= 0.5
    return np.ma.masked_array(nodal[p.tris].mean(axis=1), ~solid)


def plot_solid_field(ax, p, values, norm):
    """A masked per-triangle field on a colour scale shared across frames."""
    m = ax.tripcolor(triangulation(p), facecolors=values, cmap=style.SEQ,
                     norm=norm, rasterized=True)
    outline_domain(ax, p)
    bare(ax, p)
    return m


def plot_field(ax, p, g_tri, clip=98.0):
    """A per-triangle magnitude field, gamma-compressed and clipped."""
    vmax = np.percentile(g_tri, clip)
    m = ax.tripcolor(triangulation(p), facecolors=np.minimum(g_tri, vmax),
                     cmap=style.SEQ, norm=PowerNorm(0.45, vmin=0.0, vmax=vmax),
                     rasterized=True)
    bare(ax, p)
    return m


def plot_history(ax, hist, stop=None, symbols=True):
    """Plot only states already reached, including the initial full domain.

    `symbols=False` drops the notation from the legend, for the animation of
    the introduction, which runs before any of it has been defined.
    """
    stop = len(hist["J"]) if stop is None else stop
    it = np.arange(stop)
    ax.plot(it, hist["V"][:stop], color=style.BLUE,
            label=r"material fraction $|\Omega_k|/|\Omega_0|$" if symbols
            else "material fraction")
    ax.plot(it, hist["J"][0] / np.asarray(hist["J"][:stop]), color=style.ORANGE,
            label=r"relative stiffness $J_0/J_k$" if symbols
            else "relative stiffness")
    ax.set_xlim(0, len(hist["J"]) - 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("iteration")
    ax.legend(loc="lower left", fontsize=8)


# -----------------------------------------------------------------------------
# Boundary-condition symbols
# -----------------------------------------------------------------------------
def piers(ax, xs, base, out):
    """Pinned point supports at abscissae `xs` on the line y = `base`.

    `out` is the signed height of the symbol, pointing away from the material,
    so one call serves both piers standing under a deck and piers hanging one
    from above.
    """
    for x in xs:
        ax.add_patch(Polygon([(x, base), (x - .065, base + out),
                              (x + .065, base + out)],
                             facecolor=style.SURFACE, edgecolor=style.INK_2,
                             lw=1.3, zorder=6))
        ground = base + 1.18 * out
        ax.plot([x - .09, x + .09], [ground, ground], color=style.INK_2, lw=1)


def clamped_edge(ax, xy, tick):
    """A clamped boundary: the node polyline `xy`, hatched along `tick`."""
    ax.plot(xy[:, 0], xy[:, 1], color=style.INK_2, lw=2.5, zorder=5)
    for x, y in xy[np.linspace(0, len(xy) - 1, min(13, len(xy)), dtype=int)]:
        ax.plot([x, x + tick[0]], [y, y + tick[1]], color=style.INK_2, lw=1,
                zorder=5)


def loaded_edge(ax, xy):
    """Gamma_N itself: the node polyline carrying the prescribed traction."""
    ax.plot(xy[:, 0], xy[:, 1], color=style.ORANGE, lw=3, zorder=6)


def down_arrows(ax, xs, tail, head):
    """Downward traction arrows at `xs`, drawn from y = `tail` to y = `head`.

    They illustrate the direction of the traction only, not the unequal nodal
    quadrature weights at the endpoints of a uniformly loaded edge.
    """
    for x in xs:
        ax.annotate("", xy=(x, head), xytext=(x, tail), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", lw=1.2,
                                    color=style.ORANGE))
