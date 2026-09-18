"""Shared matplotlib styling and palette for the Feynman-Kac figures.

Same chrome as the other posts (recessive gray axes, single documented palette).
The boundary datum `g` is signed, so it takes a *diverging* encoding: two hues
with a neutral gray midpoint, equal steps per arm, blue = cold, red = hot.  The
convergence panel deliberately avoids both poles -- its estimate is ink, its
reference orange -- so that no color means two things across the two panels.
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"      # chart surface
INK = "#0b0b0b"          # primary ink
INK_2 = "#52514e"        # secondary ink
MUTED = "#898781"        # axis / tick labels
GRID = "#e1e0d9"         # hairline gridline
BASELINE = "#c3c2b7"     # baseline / axis

BLUE = "#2a78d6"         # categorical slot 1
ORANGE = "#eb6834"       # categorical slot 2
RED = "#e34948"          # categorical slot 8

# Diverging ramp for the boundary temperature: blue arm <-> gray <-> red arm.
COLD = ["#0d366b", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"]
NEUTRAL = "#f0efec"
HOT = ["#fbd9d9", "#f5aeae", "#ed7c7b", "#e34948", "#c33534", "#9d2524", "#771a19"]
DIV = LinearSegmentedColormap.from_list("div_blue_red", COLD + [NEUTRAL] + HOT)

# Sequential blue ramp, steps 100 -> 700 (light = near zero).
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

PATH = "#52514e"         # a simulated trajectory: ink, never a data hue

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
