"""Shared matplotlib styling and palette for the topological-derivative figures.

Single-hue blue sequential ramp for magnitude fields, a two-tone (ink / surface)
encoding for the binary material indicator, and recessive gray chrome.  All
values come from one documented palette so every figure in the post reads as
one system.
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"      # chart surface
INK = "#0b0b0b"          # primary ink
INK_2 = "#52514e"        # secondary ink
MUTED = "#898781"        # axis / tick labels
GRID = "#e1e0d9"         # hairline gridline

BLUE = "#2a78d6"         # categorical slot 1
ORANGE = "#eb6834"       # categorical slot 2
RED = "#e34948"          # categorical slot 8

# Sequential blue ramp, steps 100 -> 700 (light = near zero).
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

# Material / void: solid material is dark blue ink, void is the bare surface.
SOLID = "#1c5cab"
VOID = SURFACE

RC = {
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 4.5,
}


def use():
    mpl.rcParams.update(RC)
