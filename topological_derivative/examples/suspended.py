"""Suspended deck: the hanging bridge's load, hung from three piers above it."""

import style
from .deck import DeckBridge


class SuspendedBridge(DeckBridge):
    """The exact vertical mirror of `Bridge`.

    Reflecting y -> ly - y flips the parity of i + j, which is precisely what
    maps a union-jack diagonal onto the one its mirror needs, so the reflected
    triangulation is the original.  Both J = f.u and the topological derivative
    are even in the sign of the state, so one run at the baseline settings is
    enough - a sweep would reproduce td_bridge_sweep.png upside down.
    """

    name = "suspended"
    title = "Suspended deck"
    label = "three piers on the top edge, deck hung underneath"
    support_condition = "three pins on the top edge: ux=uy=0 at x=0, L/2, L"
    cache_tag = "_3pier_hung"

    pier_x = (0.0, 0.5, 1.0)
    pier_edge, load_edge = "top", "bottom"

    ylim = (-0.42, 1.42)
    figure_file = "td_suspended.png"
    figure_title = ("Suspended deck: the same load hung from three piers on "
                    "the top edge")

    def label_supports(self, ax, compact=False):
        ax.text(.5 * self.lx, self.ly + .17,
                r"$\Gamma_D:\ u_x=u_y=0$ at three piers",
                ha="center", va="bottom", fontsize=8, color=style.INK_2)
