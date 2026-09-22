"""
The uniformly loaded deck on pinned piers, shared by the three bridges.

`bridge`, `hanging` and `suspended` use the same 3 x 1 domain, mesh, material,
schedule and total load, and differ only in which long edge the piers stand on,
which long edge carries the traction, and where their labels fit.  A subclass
therefore declares `pier_x`, `pier_edge` and `load_edge`, and supplies the
Gamma_D label - the one thing that does not follow from those three.
"""

import numpy as np

import draw
import style
from problem import Problem


class DeckBridge(Problem):
    lx, ly = 3.0, 1.0
    shape = (180, 60)
    n_iter = 120

    pier_x = ()                     # pier abscissae, as fractions of lx
    pier_edge = "bottom"            # the long edge the piers stand on
    load_edge = "top"               # the long edge Gamma_N covers

    # -- geometry -------------------------------------------------------------
    @property
    def piers(self):
        """Pier abscissae in physical units."""
        return np.asarray(self.pier_x, float) * self.lx

    @property
    def piers_on_load_edge(self):
        """True when the supports sit on the loaded edge, as in `hanging`."""
        return self.pier_edge == self.load_edge

    def _edge_row(self, edge):
        """Node-grid row index of a named long edge."""
        return 0 if edge == "bottom" else self.ny

    def _band(self, Yc, edge, width):
        """Cell mask for the strip of thickness `width` along a long edge."""
        return Yc < width if edge == "bottom" else Yc > self.ly - width

    def boundary_conditions(self):
        nid = self.node_grid()
        hx, hy = self.h

        # Every pier restrains horizontal *and* vertical translation: an arch
        # can only stand if its springings take horizontal thrust.
        pins = nid[self.node_column(self.piers), self._edge_row(self.pier_edge)]
        self.fixed = self.clamp(pins)

        # Gamma_N is a whole long edge, carrying a uniform traction scaled to
        # total magnitude 1 so that J/J_0 stays comparable across load cases.
        line = nid[:, self._edge_row(self.load_edge)]
        self.f = self.pb.edge_load(self.edges_on_line(line),
                                   np.array([0.0, -1.0 / self.lx]))

        # The deck carries the traction and the pier heads receive the
        # reactions; both must stay solid or the greedy step can load pure
        # ersatz material.
        Xc, Yc = self.cell_centers()
        deck = self._band(Yc, self.load_edge, 2 * hy)
        heads = (self._band(Yc, self.pier_edge, 3 * hy) &
                 (np.abs(Xc[..., None] - self.piers).min(-1) < 3 * hx))
        self.keep = (deck | heads).ravel()

    # -- boundary-condition symbols -------------------------------------------
    def draw_supports(self, ax):
        # The piers stand under the deck, except where they hang it from above
        # and the symbols are mirrored.
        base, out = (self.ly, .11) if self.pier_edge == "top" else (0.0, -.11)
        draw.piers(ax, self.piers, base, out)
        self.label_supports(ax)

    def label_supports(self, ax):
        """Place the Gamma_D label, which is what the variants disagree on."""
        raise NotImplementedError

    def draw_load(self, ax):
        xy = self.nodes[self.loaded_nodes]
        draw.loaded_edge(ax, xy)
        # The arrows press down onto the top edge, or pull the bottom edge
        # downwards. Where a pier sits on the loaded edge its symbol takes
        # priority and the arrow beside it is dropped.
        if self.load_edge == "top":
            tail, head, label_y, va = self.ly + .22, self.ly, self.ly + .27, "baseline"
        else:
            tail, head, label_y, va = 0.0, -.24, -.27, "top"
        xs = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 13)
        if self.piers_on_load_edge:
            xs = [x for x in xs if np.abs(self.piers - x).min() >= .14]
        draw.down_arrows(ax, xs, tail, head)
        # Past the arrow tails, clear of the piers wherever they stand.
        ax.text(.5 * self.lx, label_y, r"$\Gamma_N$", ha="center", va=va,
                color=style.ORANGE, fontsize=10)
