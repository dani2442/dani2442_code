"""The figures of the post.

Four of them: one Brownian path leaving the domain (the picture the formula
is about), the harmonic extension itself (what the formula computes), the
animation of the Monte Carlo that estimates it at one point, and the exit
distribution -- the deterministic measure that the dots on the boundary are
sampling from.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgb
import matplotlib.patheffects as pe
from PIL import Image

import style
from domain import Arclength

NORM = Normalize(vmin=-1.0, vmax=1.0)   # one scale for g and for u


def _save(fig, path, dpi=170):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


# -----------------------------------------------------------------------------
# Pieces of the domain panel
# -----------------------------------------------------------------------------
def draw_boundary(ax, bean, poly, lw=3.2):
    """The boundary curve, colored by the datum it carries."""
    segs = np.stack([poly[:-1], poly[1:]], axis=1)
    lc = LineCollection(segs, cmap=style.DIV, norm=NORM, lw=lw,
                        capstyle="round", zorder=4)
    lc.set_array(bean.g(0.5 * (poly[:-1] + poly[1:])))
    ax.add_collection(lc)
    return lc


def draw_domain(ax, bean, poly, fill="#f4f3f0"):
    ax.fill(poly[:, 0], poly[:, 1], color=fill, zorder=1, lw=0)
    draw_boundary(ax, bean, poly)
    x0, x1, y0, y1 = bean.bbox
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()


HALO = [pe.withStroke(linewidth=3.0, foreground=style.SURFACE)]


def mark_start(ax, x0, label=r"$x_0$", dy=0.07):
    ax.plot(*x0, "o", ms=6.5, mfc=style.INK, mec=style.SURFACE, mew=1.4, zorder=8)
    ax.annotate(label, x0, xytext=(x0[0], x0[1] + dy), ha="center", va="bottom",
                color=style.INK, fontsize=11, zorder=8, path_effects=HALO)


# -----------------------------------------------------------------------------
# 0. The opening picture: one path, one exit
# -----------------------------------------------------------------------------
def draw_plain_domain(ax, bean, poly, fill="#f4f3f0"):
    """The bean carrying no data -- the opening figure comes before `g` exists."""
    ax.fill(poly[:, 0], poly[:, 1], color=fill, zorder=1, lw=0)
    ax.plot(poly[:, 0], poly[:, 1], color=style.INK_2, lw=2.4, zorder=4,
            solid_joinstyle="round")
    x0, x1, y0, y1 = bean.bbox
    ax.set_xlim(x0 - 0.04, x1 + 0.04)
    ax.set_ylim(y0 - 0.04, y1 + 0.04)
    ax.set_aspect("equal")
    ax.set_axis_off()


def figure_walk(bean, x0, trace, exit_pt, out, name="fk_bean_walk.gif",
                n_frames=66, n_hold=10, stride=5, omega_at=(-0.70, -0.34),
                d_omega_at=(0.80, -0.55)):
    """One Brownian path leaving the bean: the picture behind $u(x)=E[g(B_tau)]$.

    Deliberately bare -- the domain, the starting point, the path, the exit
    point, and nothing else.  The boundary datum has not been introduced at the
    point in the post where this figure appears, so the boundary is drawn in a
    single neutral colour rather than coloured by `g`.
    """
    poly = bean.boundary(900)
    t = trace[::stride]
    if not np.allclose(t[-1], trace[-1]):
        t = np.vstack([t, trace[-1]])
    cuts = np.unique(np.linspace(2, len(t), n_frames).astype(int))
    palette = _gif_palette()

    # Small figure, high dpi: same pixel size as before, but every point-sized
    # thing (labels, markers, the path) is much larger relative to the bean.
    fig = plt.figure(figsize=(3.85, 2.33), dpi=214)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])

    images = []
    for f, m in enumerate(list(cuts) + [len(t)] * n_hold):
        done = f >= len(cuts) - 1
        ax.clear()
        draw_plain_domain(ax, bean, poly)
        ax.annotate(r"$\Omega$", omega_at, ha="center", va="center",
                    fontsize=15, color=style.MUTED, zorder=3)
        # Each label wears the colour of the thing it names: the boundary curve,
        # the start dot, the exit dot.
        ax.annotate(r"$\partial\Omega$", d_omega_at, ha="center", va="center",
                    fontsize=14, color=style.INK_2, zorder=6,
                    path_effects=HALO)
        ax.plot(t[:m, 0], t[:m, 1], lw=0.85, color=style.PATH, alpha=0.9,
                zorder=5, solid_joinstyle="round")
        # The path doubles back over the start, so the label needs the halo to
        # stay readable where the tangle is densest.  Start and exit share one
        # colour: they are the two ends of the same object, $B_0 = x$ and
        # $B_\tau$.
        ax.plot(*x0, "o", ms=7.0, mfc=style.ORANGE, mec=style.SURFACE, mew=1.5,
                zorder=8)
        ax.annotate(r"$x$", x0, xytext=(-13, -11), textcoords="offset points",
                    ha="right", va="top", color=style.ORANGE, fontsize=14,
                    zorder=9, path_effects=HALO)
        if done:
            ax.plot(*exit_pt, "o", ms=7.5, mfc=style.ORANGE,
                    mec=style.SURFACE, mew=1.5, zorder=8)
            ax.annotate(r"$B_\tau$", exit_pt, xytext=(-15, 1),
                        textcoords="offset points", ha="right", va="center",
                        color=style.ORANGE, fontsize=14, zorder=9,
                        path_effects=HALO)
        else:
            # The tip is the one moving thing in the frame, and it is the same
            # orange it will keep once it lands on the boundary.
            ax.plot(t[m - 1, 0], t[m - 1, 1], "o", ms=5.5, mfc=style.ORANGE,
                    mec=style.SURFACE, mew=1.2, zorder=7)

        fig.canvas.draw()
        images.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        ).quantize(palette=palette, dither=Image.Dither.NONE))

    # A single, uniform rate for the walk itself: an opening slow-down read as
    # a stutter rather than as an easing-in.
    durations = [70] * len(images)
    durations[-n_hold:] = [90] * n_hold
    durations[-1] = 2800
    images[0].save(out / name, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False, disposal=2)
    plt.close(fig)
    print(f"wrote {name} ({len(images)} frames)")


# -----------------------------------------------------------------------------
# 0b. The same frame, solved: the FEM field on its mesh
# -----------------------------------------------------------------------------
def figure_fem(bean, pts, tri, u, x0, out, name="fk_bean_fem.png",
               figsize=(3.85, 2.33), dpi=214, omega_at=(-0.70, -0.34),
               d_omega_at=(0.80, -0.55)):
    """The finite element solution on its triangulation, labelled as the GIF is.

    Deliberately the same frame, figure size and dpi as `figure_walk`, so the
    two sit side by side in the post at the same pixel size: the same bean, once
    with a single path crossing it and once with the field that path is
    estimating.  The three marks the animation carries are carried here too, at
    the same positions -- `Omega`, `dOmega`, and the starting point `x` in its
    orange -- and nothing else: no colourbar, no title, no path.  The diverging
    scale is the one the rest of the post uses (`NORM`), so the colours are
    already calibrated by the panel that does carry a colourbar.
    """
    poly = bean.boundary(900)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])

    ax.tripcolor(pts[:, 0], pts[:, 1], tri, u, cmap=style.DIV, norm=NORM,
                 shading="gouraud", zorder=2, rasterized=True)
    ax.triplot(pts[:, 0], pts[:, 1], tri, color=style.INK, lw=0.3, alpha=0.28,
               zorder=3)
    ax.plot(poly[:, 0], poly[:, 1], color=style.INK_2, lw=2.4, zorder=4,
            solid_joinstyle="round")

    # `Omega` sits on the cold end of the ramp, which is far too dark to take
    # the GIF's gray, so it is the one mark whose colour has to change: surface
    # white against the deep blue.  `dOmega` labels the curve from outside the
    # bean, on the same background as in the GIF, so it keeps its ink.
    ax.annotate(r"$\Omega$", omega_at, ha="center", va="center",
                fontsize=15, color=style.SURFACE, zorder=6)
    ax.annotate(r"$\partial\Omega$", d_omega_at, ha="center", va="center",
                fontsize=14, color=style.INK_2, zorder=6, path_effects=HALO)
    ax.plot(*x0, "o", ms=7.0, mfc=style.ORANGE, mec=style.SURFACE, mew=1.5,
            zorder=8)
    ax.annotate(r"$x$", x0, xytext=(-13, -11), textcoords="offset points",
                ha="right", va="top", color=style.ORANGE, fontsize=14,
                zorder=9, path_effects=HALO)

    bx0, bx1, by0, by1 = bean.bbox
    ax.set_xlim(bx0 - 0.04, bx1 + 0.04)
    ax.set_ylim(by0 - 0.04, by1 + 0.04)
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.canvas.draw()
    Image.fromarray(
        np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).save(out / name)
    plt.close(fig)
    print(f"wrote {name}")


# -----------------------------------------------------------------------------
# 1. The solution
# -----------------------------------------------------------------------------
def figure_field(bean, ref, x0, u0, out):
    """The harmonic extension of g, with the level sets of u."""
    poly = bean.boundary(900)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    draw_domain(ax, bean, poly)
    m = ax.contourf(ref.X, ref.Y, ref.field, levels=np.linspace(-1, 1, 81),
                    cmap=style.DIV, norm=NORM, zorder=2, extend="neither")
    ax.contour(ref.X, ref.Y, ref.field, levels=np.arange(-0.9, 0.95, 0.15),
               colors=[style.INK], linewidths=0.45, alpha=0.30, zorder=3)
    mark_start(ax, x0)
    ax.annotate(f"$u(x_0) = {u0:+.4f}$", x0, xytext=(x0[0], x0[1] - 0.10),
                ha="center", va="top", color=style.INK, fontsize=10.5, zorder=8,
                path_effects=HALO)
    cb = fig.colorbar(m, ax=ax, fraction=0.030, pad=0.02,
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label(r"$u$  in $\Omega$,   $g$  on $\partial\Omega$", color=style.INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=style.MUTED, labelcolor=style.INK_2, length=0)
    ax.set_title(r"$\Delta u = 0$ in $\Omega$,  $u = g$ on $\partial\Omega$"
                 "\n" r"level sets every $0.15$", color=style.INK, pad=6)
    fig.tight_layout()
    _save(fig, out / "fk_bean_field.png")


# -----------------------------------------------------------------------------
# 2. The exit distribution
# -----------------------------------------------------------------------------
def figure_measure(bean, ref, x0, exits, out, nbin=64):
    """Where the paths land, against the harmonic measure of x0."""
    poly = bean.boundary(900)
    al = Arclength(poly)
    edges = np.linspace(0, al.total, nbin + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]

    w, _ = ref.weights(x0)
    p_fd = np.histogram(al.project(ref.bnd_xy), bins=edges, weights=w)[0] / width
    p_mc = np.histogram(al.project(exits), bins=edges)[0] / (len(exits) * width)

    fig, (ax, cax) = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[10, 1],
                                                   hspace=0.08))
    ax.bar(mid, p_mc, width=0.92 * width, color=style.GRID, lw=0,
           label=f"{len(exits):,} simulated exits", zorder=2)
    ax.step(np.r_[edges[0], mid, edges[-1]], np.r_[p_fd[0], p_fd, p_fd[-1]],
            where="mid", color=style.INK, lw=1.8, zorder=3,
            label=r"harmonic measure $\omega_{x_0}$  (linear solve)")
    s_star = float(al.project(x0)[0])
    ax.axvline(s_star, color=style.MUTED, lw=0.9, ls=":", zorder=1)
    ax.annotate("boundary point\nnearest $x_0$",
                (s_star, 0.72 * ax.get_ylim()[1]), xytext=(-10, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=9, color=style.INK_2)
    ax.set_ylabel("density w.r.t. arclength")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(r"The exit distribution of $B$ started at $x_0$",
                 color=style.INK, pad=6)
    ax.grid(axis="x", visible=False)

    # A strip of the boundary datum, so a peak can be read against the data.
    s = al.cum
    cax.pcolormesh(s, [0, 1], bean.g(poly)[None, :-1], cmap=style.DIV, norm=NORM,
                   shading="flat", rasterized=True)
    cax.set_yticks([])
    cax.set_xlabel(r"arclength along $\partial\Omega$")
    cax.set_ylabel(r"$g$", rotation=0, labelpad=10, va="center")
    cax.grid(False)
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    _save(fig, out / "fk_exit_distribution.png")


# -----------------------------------------------------------------------------
# 3. The animation
# -----------------------------------------------------------------------------
def _gif_palette():
    """One fixed 256-colour palette, so the GIF background never shimmers."""
    colors = [to_rgb(c) for c in (style.SURFACE, style.INK, style.INK_2,
                                  style.ORANGE, style.GRID, style.BASELINE,
                                  style.PATH, "#f4f3f0")]
    colors += [tuple(c[:3]) for c in style.DIV(np.linspace(0, 1, 176))]
    colors += [(x, x, x) for x in np.linspace(0, 1, 64)]
    colors += [to_rgb(style.SURFACE)] * (256 - len(colors))
    palette = Image.new("P", (1, 1))
    palette.putpalette(
        np.rint(255 * np.asarray(colors)).astype("uint8").ravel().tolist())
    return palette


def figure_animation(bean, x0, u0, exits, traces, out, name="fk_bean_mc.gif",
                     n_frames=80, stride=20):
    """Paths accumulating on the left, the estimate they produce on the right."""
    poly = bean.boundary(900)
    v = bean.g(exits)
    n = len(v)

    # Exit points sit on a curve, so 6,000 of them drawn in place would just
    # re-draw the boundary.  Pushed outward along the normal by a fixed random
    # amount (fixed, so the cloud does not shimmer between frames), the same
    # points become a band whose local darkness is the exit density.
    nrm = bean.grad_phi(exits)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    off = exits + nrm * (0.018 + 0.038 * np.random.default_rng(1).random((n, 1)))
    k = np.arange(1, n + 1)
    mean = np.cumsum(v) / k
    var = np.maximum(np.cumsum(v ** 2) / k - mean ** 2, 0.0) * k / np.maximum(k - 1, 1)
    half = 1.96 * np.sqrt(var / k)

    counts = np.unique(np.round(np.logspace(0, np.log10(n), n_frames)).astype(int))
    palette = _gif_palette()

    fig = plt.figure(figsize=(9.8, 4.15), dpi=110)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.28, 1], wspace=0.16,
                            left=0.015, right=0.975, bottom=0.17, top=0.855)
    ax_d = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    head = fig.text(0.015, 0.962, "", fontsize=11.5, ha="left", va="center",
                    color=style.INK)
    fig.text(0.015, 0.035, r"exit points are drawn just outside $\partial\Omega$,"
             " so that where the paths land is visible as a pile-up",
             fontsize=8.5, ha="left", va="center", color=style.MUTED)

    images = []
    for m in counts:
        ax_d.clear()
        ax_c.clear()
        draw_domain(ax_d, bean, poly)

        # Only the last few paths stay on screen: the point of the late frames
        # is the density of the dots, which a dozen tangled paths would bury.
        shown = min(m, len(traces))
        fade = [0.10, 0.18, 0.32, 0.75]
        for j, i in enumerate(range(max(0, shown - 4), shown)):
            t = traces[i][::stride]
            a = fade[j - (4 - min(shown, 4))]
            ax_d.plot(t[:, 0], t[:, 1], lw=0.75 if a > 0.5 else 0.5,
                      color=style.INK if a > 0.5 else style.PATH, alpha=a,
                      zorder=5 + (a > 0.5), solid_joinstyle="round")
        big = m < 80
        ax_d.scatter(off[:m, 0], off[:m, 1], c=v[:m], cmap=style.DIV,
                     norm=NORM, s=20 if big else 7,
                     alpha=0.95 if big else 0.55,
                     lw=0.4 if big else 0.2, edgecolors=style.INK_2, zorder=7)
        for i in range(min(m, len(traces))):
            if i >= max(0, min(m, len(traces)) - 4):
                ax_d.plot([exits[i, 0], off[i, 0]], [exits[i, 1], off[i, 1]],
                          lw=0.5, color=style.MUTED, alpha=0.5, zorder=6)
        mark_start(ax_d, x0)
        ax_d.set_title(r"$B$ started at $x_0$, stopped at $\tau$;"
                       r"  each dot is one exit point $B_\tau$",
                       fontsize=10, pad=4)

        ax_c.fill_between(k[:m], mean[:m] - half[:m], mean[:m] + half[:m],
                          color=style.GRID, lw=0, zorder=2,
                          label="95% interval")
        ax_c.plot(k[:m], mean[:m], color=style.INK, lw=1.8, zorder=3,
                  label=r"$\hat u_N$")
        ax_c.axhline(u0, color=style.ORANGE, lw=1.8, ls="--", zorder=4,
                     label=f"$u(x_0) = {u0:.4f}$")
        ax_c.plot([m], [mean[m - 1]], "o", ms=5.5, mfc=style.INK,
                  mec=style.SURFACE, mew=1.2, zorder=5)
        ax_c.set_xscale("log")
        ax_c.set_xlim(1, n)
        ax_c.set_ylim(u0 - 0.62, u0 + 0.62)
        ax_c.set_xlabel("paths $N$")
        ax_c.legend(loc="upper right", fontsize=9)
        ax_c.set_title(r"$\hat u_N = \frac{1}{N}\sum_{i\leq N} g(B^i_\tau)$"
                       r"$\;\longrightarrow\;\mathbb{E}_{x_0}[g(B_\tau)]$",
                       fontsize=10, pad=6)
        head.set_text(f"$N = {m:,}$ paths      "
                      f"$\\hat u_N = {mean[m - 1]:+.4f} \\pm {half[m - 1]:.4f}$"
                      f"      $u(x_0) = {u0:+.4f}$")

        fig.canvas.draw()
        images.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        ).quantize(palette=palette, dither=Image.Dither.NONE))

    durations = [150] * len(images)
    durations[:6] = [520] * 6
    durations[-1] = 2600
    images[0].save(out / name, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False, disposal=2)
    plt.close(fig)
    print(f"wrote {name} ({len(images)} frames)")
