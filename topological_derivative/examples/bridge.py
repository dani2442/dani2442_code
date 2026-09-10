"""Three-pier bridge: a loaded deck standing on two abutments and a midspan pier."""

import style
from .deck import DeckBridge


class Bridge(DeckBridge):
    """Both spans are continuous over supports that can carry horizontal thrust."""

    name = "bridge"
    title = "Three-pier bridge"
    label = "three pinned piers, uniformly loaded deck"
    support_condition = "three pins: ux=uy=0 at x=0, L/2, L"
    cache_tag = "_3pier_deck"

    pier_x = (0.0, 0.5, 1.0)
    pier_edge, load_edge = "bottom", "top"

    ylim = (-0.32, 1.36)
    gif_file = "td_bridge.gif"

    def label_supports(self, ax, compact=False):
        ax.text(.5 * self.lx, -.155, r"$\Gamma_D:\ u_x=u_y=0$ at three piers",
                ha="center", va="top", fontsize=8, color=style.INK_2)
