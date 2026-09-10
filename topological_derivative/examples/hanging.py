"""Hanging bridge: two end piers, with the loaded deck slung underneath."""

import style
from .deck import DeckBridge


class HangingBridge(DeckBridge):
    """Load and supports share the bottom edge.

    The two corner nodes are therefore both loaded and pinned, so a fraction
    hx/lx = 0.56% of the nodal load is taken straight by the supports and does
    no work.  Ratios for this case use its own J_0.
    """

    name = "hanging"
    title = "Hanging bridge"
    label = "two end piers, deck hung underneath"
    support_condition = "two pins: ux=uy=0 at x=0, L"
    cache_tag = "_2pier_slung"

    pier_x = (0.0, 1.0)
    pier_edge, load_edge = "bottom", "bottom"

    ylim = (-0.56, 1.10)
    figure_file = "td_hanging.png"
    figure_title = ("Hanging bridge: uniformly loaded deck beneath a "
                    "single-span arch")

    def label_supports(self, ax, compact=False):
        # Both boundaries sit at the bottom here, so the support labels go
        # outboard of the load arrows and on a line of their own. In a sweep
        # panel there is only room for the symbol.
        for x, ha in ((0.0, "left"), (self.lx, "right")):
            ax.text(x, -.40, r"$\Gamma_D$" if compact
                    else r"$\Gamma_D:\ u_x=u_y=0$", ha=ha, va="top",
                    fontsize=8, color=style.INK_2)
