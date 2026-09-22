"""L-bracket: a clamped vertical arm and a traction on the horizontal one."""

import numpy as np

import draw
import style
from problem import Problem


class LBracket(Problem):
    """The re-entrant corner at (cut, cut) is the point of the example."""

    name = "lbracket"
    title = "L-bracket"
    label = "L-shaped domain, re-entrant corner"

    lx, ly = 1.0, 1.0
    shape = (120, 120)
    cut = 0.4                       # the L's inner corner

    figsize = (5.2, 5.2)
    figure_file = "td_lbracket.png"

    def boundary_conditions(self):
        nid = self.node_grid()
        hx, hy = self.h
        cut = self.cut

        # A permanently void upper-right block makes the domain L-shaped.
        Xc, Yc = self.cell_centers()
        self.void = ((Xc > cut) & (Yc > cut)).ravel()

        # Clamp the part of the top edge that belongs to the vertical arm.
        top = nid[:, self.ny]
        self.fixed = self.clamp(top[self.nodes[top, 0] <= cut + 1e-12])

        # Gamma_N is horizontal: the right end of the upper face of the
        # horizontal arm, at y = cut, with a downward traction normal to it.
        line = nid[:, self.node_row(cut)]
        x = self.nodes[line, 0]
        self.half = max(0.05, 1.01 * hx)             # at least one element edge
        patch = x >= self.lx - 2 * self.half - 1e-12
        edges = self.edges_on_line(line, patch)
        length = np.abs(x[patch].max() - x[patch].min())
        self.f = self.pb.edge_load(edges, np.array([0.0, -1.0 / length]))

        self.keep = ((Yc <= cut) & (Yc > cut - 3 * hy) &
                     (Xc > self.lx - 2 * self.half - 2 * hx)).ravel()

    # -- boundary-condition symbols -------------------------------------------
    def draw_supports(self, ax):
        draw.clamped_edge(ax, self.nodes[self.fixed_nodes], tick=(.025, .025))
        ax.text(.2, 1.07, r"$\Gamma_D$", ha="center", color=style.INK_2,
                fontsize=10)

    def draw_load(self, ax):
        xy = self.nodes[self.loaded_nodes]
        draw.loaded_edge(ax, xy)
        # The loaded patch is horizontal, so the arrows press down onto it from
        # the permanently void block above.
        y0 = xy[:, 1].mean()
        draw.down_arrows(ax, np.linspace(xy[:, 0].min(), xy[:, 0].max(), 6),
                         tail=y0 + .17, head=y0)
        ax.text(xy[:, 0].mean(), y0 + .21, r"$\Gamma_N$", ha="center",
                color=style.ORANGE, fontsize=10)
