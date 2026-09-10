"""Cantilever: the whole left edge clamped, a short traction patch at the tip."""

import numpy as np

import draw
import style
from problem import Problem


class Cantilever(Problem):
    name = "cantilever"
    title = "Cantilever"
    label = "clamped left edge, tip load"

    lx, ly = 2.0, 1.0
    shape = (150, 75)

    gif_file = "td_optimization.gif"

    def boundary_conditions(self):
        nid = self.node_grid()
        hx, hy = self.h

        # Clamp the whole left edge.
        self.fixed = self.clamp(nid[0])

        # Downward traction of total magnitude 1 on a patch at the right edge.
        y = self.nodes[nid[self.nx], 1]
        self.half = max(0.06 * self.ly, 1.01 * hy)   # at least one element edge
        patch = np.abs(y - 0.5 * self.ly) <= self.half
        edges = self.edges_on_line(nid[self.nx], patch)
        length = np.abs(y[patch].max() - y[patch].min())
        self.f = self.pb.edge_load(edges, np.array([0.0, -1.0 / length]))

        # The cells behind the load patch stay solid, or the greedy step can
        # end up applying the traction to pure ersatz material.
        Xc, Yc = self.cell_centers()
        self.keep = ((Xc > self.lx - 3 * hx) &
                     (np.abs(Yc - 0.5 * self.ly) < self.half + 2 * hy)).ravel()

    # -- boundary-condition symbols -------------------------------------------
    def draw_supports(self, ax, compact=False):
        draw.clamped_edge(ax, self.nodes[self.fixed_nodes], tick=(-.025, -.025))
        ax.text(-.07, .5 * self.ly, r"$\Gamma_D$", ha="right", va="center",
                color=style.INK_2, fontsize=10)

    def draw_load(self, ax, compact=False):
        xy = self.nodes[self.loaded_nodes]
        draw.loaded_edge(ax, xy)
        # A leader identifies the loaded vertical patch; the adjacent arrow
        # shows the (tangential) downward traction without hiding the patch.
        center = xy[:, 1].mean()
        ax.plot([self.lx, self.lx + .10], [center, center], lw=.9,
                color=style.ORANGE)
        ax.annotate("", xy=(self.lx + .10, center - .13),
                    xytext=(self.lx + .10, center + .13), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", lw=1.5,
                                    color=style.ORANGE))
        ax.text(self.lx + .10, center + .16, r"$\Gamma_N$", ha="center",
                color=style.ORANGE, fontsize=10)
