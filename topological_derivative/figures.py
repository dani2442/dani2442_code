"""
The figures of the post.

Every figure takes a `Problem` and asks it to annotate its own axes, so nothing
here needs to know which load case it is rendering.  Output paths are passed
in, so preview assets can be written anywhere.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, to_rgb
from PIL import Image

import draw
import style


def _save(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


# -----------------------------------------------------------------------------
# Setup and the raw gradient
# -----------------------------------------------------------------------------
def figure_mesh(p, out, run_shape):
    """The triangulated design domain and its boundary conditions."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.triplot(draw.triangulation(p), lw=0.55, color=style.MUTED, alpha=0.55)
    p.annotate(ax)
    draw.bare(ax, p)
    ax.set_title(r"Design domain $\Omega_0$, shown at $%d\times%d$ cells "
                 r"(%d nodes, %d triangles)"
                 % (p.nx, p.ny, len(p.nodes), len(p.tris)) + "\n"
                 r"the runs below use $%d\times%d$ cells" % run_shape,
                 color=style.INK, fontsize=10)
    fig.tight_layout()
    _save(fig, out / "td_mesh.png", dpi=170)


def figure_gradient(p, g_tri, out):
    """The topological gradient on the full-material domain."""
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    m = draw.plot_field(ax, p, g_tri)
    p.annotate(ax)
    cb = fig.colorbar(m, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label(r"$D_TJ(\hat x)$", color=style.INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=style.MUTED, labelcolor=style.INK_2)
    ax.set_title(r"Topological gradient of the compliance on $\Omega_0$"
                 "\n" r"dark $=$ expensive to perforate,  light $=$ nearly free",
                 color=style.INK)
    fig.tight_layout()
    _save(fig, out / "td_gradient.png", dpi=170)


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
def figure_convergence(p, hist, out):
    """Static version of the history panel of the animation."""
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    draw.plot_history(ax, hist)
    ax.set_title(f"{p.title}: material use and stiffness")
    fig.tight_layout()
    _save(fig, out / "td_convergence.png", dpi=170)


def figure_example(p, chi, hist, out):
    """One load case's final design, with its actual constraints."""
    fig, ax = plt.subplots(figsize=p.figsize)
    draw.plot_design(ax, p, chi)
    p.annotate(ax)
    ax.set_title(p.figure_title + "\n" +
                 r"$V=%.2f$,  $r_{\min}/h=%.1f$,  $J/J_0=%.2f$  (%d iterations)"
                 % (hist["V"][-1], p.rmin_cells,
                    hist["J"][-1] / hist["J"][0], len(hist["J"]) - 1),
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out / p.figure_file, dpi=170)


def figure_sweep(results, out, filename, subtitle):
    """Rows change the length scale, columns change the material budget."""
    radii = sorted({p.rmin_cells for p, _, _ in results})
    volumes = sorted({p.vol_frac for p, _, _ in results})
    fig, axes = plt.subplots(len(radii), len(volumes), figsize=(13.6, 8.5),
                             squeeze=False)
    for p, chi, hist in results:
        ax = axes[radii.index(p.rmin_cells), volumes.index(p.vol_frac)]
        draw.plot_design(ax, p, chi)
        p.annotate(ax, compact=True)
        ax.set_title(r"$V=%.2f$   $r_{\min}/h=%.1f$" % (p.vol_frac, p.rmin_cells)
                     + "\n" + r"$J/J_0=%.2f$   stiffness $=%.0f\%%$"
                     % (hist["J"][-1] / hist["J"][0],
                        100 * hist["J"][0] / hist["J"][-1]), fontsize=10)
    p0, hist0 = results[0][0], results[0][2]
    fig.suptitle("One bridge load case, nine final configurations\n"
                 r"same $%d\times%d$ cell mesh, " % (p0.nx, p0.ny) + subtitle +
                 r", material and %d iterations; evolution rate $%g\%%$"
                 % (len(hist0["J"]) - 1, 100 * p0.evol_rate), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=3.2, w_pad=2)
    _save(fig, out / filename, dpi=150)


# -----------------------------------------------------------------------------
# The animation
# -----------------------------------------------------------------------------
def _gif_palette():
    """One fixed 256-colour palette, so the GIF background never shimmers.

    The figure chrome comes first, then the sequential ramp, then a gray wedge
    for the antialiased text.
    """
    colors = [to_rgb(c) for c in (style.SURFACE, style.INK, style.INK_2,
                                  style.ORANGE, style.BLUE, style.SOLID,
                                  style.GRID)]
    colors += [tuple(c[:3]) for c in style.SEQ(np.linspace(0, 1, 160))]
    colors += [(x, x, x) for x in np.linspace(0, 1, 64)]
    colors += [to_rgb(style.SURFACE)] * (256 - len(colors))
    palette = Image.new("P", (1, 1))
    palette.putpalette(
        np.rint(255 * np.asarray(colors)).astype("uint8").ravel().tolist())
    return palette


def figure_animation(p, hist, out):
    """Render every solved iteration, with a fixed colour scale across frames."""
    # The two panels have equal aspect, so their height follows from the domain:
    # fixing it, rather than the figure height, is what keeps every load case
    # free of dead space above and below the design.
    W, LEFT, RIGHT, WSPACE = 9.6, .05, .97, .12
    HIST, GAP, HEAD, FOOT = 1.20, .34, .68, .52    # inches of chrome
    panel_w = (RIGHT - LEFT) * W / (2 + WSPACE)
    aspect = (p.lx + .36 * p.ly) / ((p.ylim[1] - p.ylim[0]) * p.ly)
    panel_h = panel_w / aspect
    H = panel_h + HIST + GAP + HEAD + FOOT

    fig = plt.figure(figsize=(W, H), dpi=110)
    grid = fig.add_gridspec(2, 2, height_ratios=[panel_h, HIST], wspace=WSPACE,
                            hspace=2 * GAP / (panel_h + HIST))
    design_ax, field_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    history_ax = fig.add_subplot(grid[1, :])
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=FOOT / H, top=1 - HEAD / H)
    # One header line: the load case and the iteration on the left, the two
    # numbers the run is judged by on the right.
    head = fig.text(LEFT, 1 - .20 / H, "", fontsize=12, ha="left", va="center")
    stats = fig.text(RIGHT, 1 - .20 / H, "", fontsize=11, ha="right",
                     va="center", color=style.INK_2)

    # How far the material moves under the load, on one scale for the whole run:
    # the deflection grows as material leaves, and a per-frame scale would hide
    # exactly that.  A percentile rather than the maximum, because a single
    # intermediate state can leave a barely connected fragment whose ersatz
    # link lets it move far more than the structure ever does; that one frame
    # would otherwise flatten the scale of all the others.
    fields = [draw.solid_field(p, chi, u)
              for chi, u in zip(hist["chi"], hist["U"])]
    vmax = max(np.percentile(f.compressed(), 99.0) for f in fields)
    norm = Normalize(vmin=0, vmax=vmax, clip=True)
    palette = _gif_palette()

    images = []
    for k, (chi, field, J, V) in enumerate(zip(hist["chi"], fields, hist["J"],
                                               hist["V"])):
        design_ax.clear()
        field_ax.clear()
        history_ax.clear()
        draw.plot_design(design_ax, p, chi)
        m = draw.plot_solid_field(field_ax, p, field, norm)
        for ax in (design_ax, field_ax):
            p.annotate(ax)
        design_ax.set_title("Material on the fixed triangular mesh",
                            fontsize=10, pad=4)
        field_ax.set_title(r"Displacement magnitude $|u|$", fontsize=10, pad=4)
        draw.plot_history(history_ax, hist, k + 1, symbols=False)
        history_ax.axvline(k, color=style.MUTED, lw=.8, ls=":")
        head.set_text(f"{p.title}  |  iteration {k:02d}/{len(fields) - 1}")
        stats.set_text(f"material {V:.0%}     stiffness "
                       f"{hist['J'][0] / J:.1%}     peak $|u|$ "
                       f"{field.max():.3g}")
        if k == 0:
            # Anchored to the grid cell, not to the axes box: equal aspect
            # shrinks the latter by an amount that depends on the domain.
            box = grid[0, 1].get_position(fig)
            cax = fig.add_axes([box.x0 + .116 * box.width, box.y1 + .18 / H,
                                .693 * box.width, .055 / H])
            cb = fig.colorbar(m, cax=cax, orientation="horizontal")
            cb.set_ticks([0, vmax], labels=["0", f"{vmax:.3g}"])
            cb.ax.tick_params(length=0, labelsize=8)
            cb.outline.set_visible(False)
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        images.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))

    durations = [130] * len(images)
    durations[0], durations[-1] = 900, 2400
    images[0].save(out / p.gif_file, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False, disposal=2)
    plt.close(fig)
    print(f"wrote {p.gif_file} ({len(images)} solved states)")
