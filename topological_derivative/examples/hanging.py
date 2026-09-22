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

    ylim = (-0.40, 1.10)
    figure_file = "td_hanging.png"

    def label_supports(self, ax):
        # Both boundaries sit at the bottom here, so the labels go outboard of
        # the load arrows, one at each pier.
        for x, ha in ((0.0, "left"), (self.lx, "right")):
            ax.text(x, -.27, r"$\Gamma_D$", ha=ha, va="top", fontsize=10,
                    color=style.INK_2)
