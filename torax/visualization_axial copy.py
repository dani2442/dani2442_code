import argparse
import pathlib

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize


def ring_weights_from_profile(rho: np.ndarray, values: np.ndarray, rings: int) -> np.ndarray:
    """Bin a 1D radial profile into per-ring weights (simple mean per bin)."""
    rho = np.asarray(rho, dtype=float)
    values = np.asarray(values, dtype=float)
    if rings <= 0:
        raise ValueError("--rings must be > 0")
    if rho.ndim != 1 or values.ndim != 1 or rho.size != values.size:
        raise ValueError("Expected 1D rho and values of same length")
    if rho.size == 0:
        return np.ones(rings, dtype=float)

    order = np.argsort(rho)
    rho = rho[order]
    values = values[order]

    lo = float(np.nanmin(rho))
    hi = float(np.nanmax(rho))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones(rings, dtype=float)

    bins = np.linspace(lo, hi, rings + 1)
    out = np.zeros(rings, dtype=float)
    for i in range(rings):
        left = bins[i]
        right = bins[i + 1]
        mask = (rho >= left) & (rho < right) if i < rings - 1 else (rho >= left) & (rho <= right)
        if np.any(mask):
            out[i] = float(np.nanmean(values[mask]))
        else:
            center = 0.5 * (left + right)
            j = int(np.argmin(np.abs(rho - center)))
            out[i] = float(values[j])

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, 1e-12, None)


def _normalize_01(values: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] with safe handling for constants/non-finite."""
    v = np.asarray(values, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(np.min(v)) if v.size else 0.0
    vmax = float(np.max(v)) if v.size else 0.0
    den = vmax - vmin
    if not np.isfinite(den) or den <= 0.0:
        return np.zeros_like(v, dtype=float)
    return np.clip((v - vmin) / den, 0.0, 1.0)


def _normalize_range_01(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize to [0, 1] using explicit range (safe for constants/non-finite)."""
    v = np.asarray(values, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    den = float(vmax) - float(vmin)
    if not np.isfinite(den) or den <= 0.0:
        return np.zeros_like(v, dtype=float)
    return np.clip((v - float(vmin)) / den, 0.0, 1.0)


def _precompute_ring_points(
    *,
    r0: float,
    r_inner: float,
    r_outer: float,
    rings: int,
    dots_max: int,
    seed: int,
) -> list[np.ndarray]:
    """Pre-generate a fixed bank of random points per ring.

    Using a fixed point bank avoids frame-to-frame flicker; per frame we just
    reveal more/less of each ring's bank depending on density.
    """

    if rings <= 0:
        raise ValueError("rings must be > 0")
    if dots_max <= 0:
        raise ValueError("dots_max must be > 0")
    if r_outer <= r_inner:
        raise ValueError("r_outer must be > r_inner")

    rng = np.random.default_rng(int(seed))
    dr = (float(r_outer) - float(r_inner)) / float(rings)
    banks: list[np.ndarray] = []
    for i in range(rings):
        rin = float(r_inner) + float(i) * dr
        rout = rin + dr

        # Uniform in area within annulus: sample r^2 uniformly.
        u = rng.random(int(dots_max), dtype=float)
        r = np.sqrt(u * (rout * rout - rin * rin) + rin * rin)
        theta = rng.random(int(dots_max), dtype=float) * (2.0 * np.pi)

        x = float(r0) + r * np.cos(theta)
        y = r * np.sin(theta)
        banks.append(np.stack([x, y], axis=1))

    return banks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Axial/poloidal cross-section animation: onion-like layers with density(alpha) + temperature(color)."
    )
    parser.add_argument("--iter-nc", type=pathlib.Path, default=pathlib.Path("torax/iter.nc"))
    parser.add_argument(
        "--iter-var",
        choices=["n_e", "n_i"],
        default="n_e",
        help="Profile used for layer alpha (density).",
    )
    parser.add_argument("--temp-var", type=str, default="T_e", help="Profile used for layer colormap (temperature).")
    parser.add_argument("--cmap", type=str, default="inferno", help="Matplotlib colormap for temperature")
    parser.add_argument("--temp-vmin", type=float, default=None, help="Colorbar/colormap min for temperature")
    parser.add_argument("--temp-vmax", type=float, default=None, help="Colorbar/colormap max for temperature")
    parser.add_argument("--alpha-min", type=float, default=0.10)
    parser.add_argument("--alpha-max", type=float, default=0.95)

    parser.add_argument(
        "--dots-min",
        type=int,
        default=20,
        help="Minimum dots per ring (at lowest density).",
    )
    parser.add_argument(
        "--dots-max",
        type=int,
        default=400,
        help="Maximum dots per ring (at highest density).",
    )
    parser.add_argument(
        "--dot-size",
        type=float,
        default=15.0,
        help="Matplotlib scatter size (points^2).",
    )
    parser.add_argument(
        "--dot-lifetime",
        type=float,
        default=1.2,
        help="Seconds a dot remains visible before fading out.",
    )
    parser.add_argument(
        "--layer-alpha",
        type=float,
        default=0.14,
        help="Alpha for the onion layer fills (temperature-colored).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the dot locations (keeps dots stable across frames).",
    )

    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("torax/axial_layers.gif"))
    parser.add_argument("--rings", type=int, default=16)
    parser.add_argument("--r-inner", type=float, default=0.0, help="Inner minor radius (0 makes a filled core).")
    parser.add_argument("--r-outer", type=float, default=1.0, help="Outer minor radius of the plasma cross-section.")
    parser.add_argument(
        "--major-radius",
        type=float,
        default=1.8,
        help="Major radius (R0): centers the onion layers at (R0, 0).",
    )

    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    dt = xr.open_datatree(args.iter_nc)
    profiles = dt["profiles"]
    if args.iter_var not in profiles:
        raise KeyError(
            f"Missing profiles[{args.iter_var!r}] in {args.iter_nc}; available: {list(profiles.data_vars)}"
        )
    if args.temp_var not in profiles:
        raise KeyError(
            f"Missing profiles[{args.temp_var!r}] in {args.iter_nc}; available: {list(profiles.data_vars)}"
        )

    da_density = profiles[args.iter_var].load()  # dims: (time, rho_norm)
    da_temp = profiles[args.temp_var].load()

    time_dim = da_density.dims[0]
    rho_dim = da_density.dims[1]
    time = np.asarray(da_density.coords[time_dim].values, dtype=float)
    rho = np.asarray(da_density.coords.get("rho_norm", da_density.coords[rho_dim]).values, dtype=float)

    density_arr = np.asarray(da_density.values, dtype=float)
    temp_arr = np.asarray(da_temp.values, dtype=float)

    if args.temp_vmin is None:
        temp_vmin = float(np.nanmin(temp_arr))
    else:
        temp_vmin = float(args.temp_vmin)
    if args.temp_vmax is None:
        temp_vmax = float(np.nanmax(temp_arr))
    else:
        temp_vmax = float(args.temp_vmax)
    if not np.isfinite(temp_vmin) or not np.isfinite(temp_vmax) or temp_vmax <= temp_vmin:
        temp_vmin, temp_vmax = 0.0, 1.0

    out_path: pathlib.Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rings = int(args.rings)
    r_inner = float(args.r_inner)
    r_outer = float(args.r_outer)
    if rings <= 0:
        raise ValueError("--rings must be > 0")
    if r_outer <= r_inner:
        raise ValueError("--r-outer must be > --r-inner")

    dots_min = int(args.dots_min)
    dots_max = int(args.dots_max)
    if dots_min < 0:
        raise ValueError("--dots-min must be >= 0")
    if dots_max <= 0:
        raise ValueError("--dots-max must be > 0")
    if dots_max < dots_min:
        raise ValueError("--dots-max must be >= --dots-min")

    dr = (r_outer - r_inner) / float(rings)
    frames = int(np.ceil(float(args.seconds) * float(args.fps)))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=0.88, bottom=0, top=1)
    ax.set_aspect("equal", adjustable="box")

    R0 = float(args.major_radius)
    ax.set_xlim(R0 - r_outer * 1.12, R0 + r_outer * 1.12)
    ax.set_ylim(-r_outer * 1.12, r_outer * 1.12)

    cmap = matplotlib.colormaps.get_cmap(str(args.cmap))
    norm = Normalize(vmin=temp_vmin, vmax=temp_vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(str(args.temp_var), color="white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=10,
        zorder=10,
    )

    # Pre-generate stable dot locations for each ring; density controls how
    # many of those dots are shown per ring.
    ring_point_banks = _precompute_ring_points(
        r0=R0,
        r_inner=r_inner,
        r_outer=r_outer,
        rings=rings,
        dots_max=dots_max,
        seed=int(args.seed),
    )

    # Create onion-like annular wedges (full 360°) centered at (R0, 0).
    # We draw from outer -> inner so inner rings are on top.
    ring_patches: list[patches.Wedge] = []
    for i in range(rings - 1, -1, -1):
        outer_r = r_inner + float(i + 1) * dr
        wedge = patches.Wedge(
            center=(R0, 0.0),
            r=outer_r,
            theta1=0.0,
            theta2=360.0,
            width=dr,
            facecolor=(0, 0, 0, 0),
            edgecolor=(1, 1, 1, 0.18),
            linewidth=0.6,
            zorder=2,
        )
        ax.add_patch(wedge)
        ring_patches.append(wedge)

    dots = ax.scatter(
        [],
        [],
        s=float(args.dot_size),
        marker=".",
        linewidths=0,
        edgecolors="none",
        facecolors=np.empty((0, 4), dtype=float),
        zorder=3,
    )

    # Particle state: offsets (N,2), ring index (N,), age in frames (N,)
    particle_offsets = np.empty((0, 2), dtype=float)
    particle_ring_idx = np.empty((0,), dtype=int)
    particle_age = np.empty((0,), dtype=int)
    particle_rng = np.random.default_rng(int(args.seed) + 12345)

    lifetime_frames = int(max(1, round(float(args.dot_lifetime) * float(args.fps))))
    layer_alpha = float(np.clip(float(args.layer_alpha), 0.0, 1.0))

    # Boundary outlines.
    ax.add_patch(
        patches.Circle((R0, 0.0), radius=r_outer, fill=False, linewidth=1.2, edgecolor="white", zorder=5)
    )
    if r_inner > 0.0:
        ax.add_patch(
            patches.Circle((R0, 0.0), radius=r_inner, fill=False, linewidth=1.2, edgecolor="white", zorder=5)
        )

    def _interp_profiles(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        t_len = int(density_arr.shape[0])
        if t_len <= 1:
            return density_arr[0], temp_arr[0]
        denom = max(1, int(frames) - 1)
        pos = (float(frame_index % frames) / float(denom)) * float(t_len - 1)
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, t_len - 1)
        w = float(pos - float(i0))
        prof_d = (1.0 - w) * density_arr[i0] + w * density_arr[i1]
        prof_t = (1.0 - w) * temp_arr[i0] + w * temp_arr[i1]
        return prof_d, prof_t

    def init():
        time_text.set_text("")
        for p in ring_patches:
            p.set_facecolor((0, 0, 0, 0))
        dots.set_offsets(np.empty((0, 2), dtype=float))
        dots.set_facecolors(np.empty((0, 4), dtype=float))
        dots.set_sizes(np.empty((0,), dtype=float))

        nonlocal particle_offsets, particle_ring_idx, particle_age
        particle_offsets = np.empty((0, 2), dtype=float)
        particle_ring_idx = np.empty((0,), dtype=int)
        particle_age = np.empty((0,), dtype=int)
        return (*ring_patches, dots, time_text)

    def update(frame: int):
        t_len = int(density_arr.shape[0])
        denom = max(1, int(frames) - 1)
        pos = (float(int(frame) % frames) / float(denom)) * float(max(0, t_len - 1))
        i0 = int(np.floor(pos)) if t_len > 0 else 0
        i1 = min(i0 + 1, max(0, t_len - 1)) if t_len > 0 else 0
        w = float(pos - float(i0)) if t_len > 1 else 0.0

        t_sim = float(int(frame)) / float(args.fps)
        if time.size == t_len and t_len > 0 and np.all(np.isfinite(time)):
            t_sim = float((1.0 - w) * time[i0] + w * time[i1])
        time_text.set_text(f"t = {t_sim:.3f} s")

        prof_density, prof_temp = _interp_profiles(int(frame))
        density_rings = ring_weights_from_profile(rho, np.asarray(prof_density, dtype=float), rings)
        temp_rings = ring_weights_from_profile(rho, np.asarray(prof_temp, dtype=float), rings)

        # Temperature -> ring fill color (layers remain visible).
        t01 = _normalize_range_01(temp_rings, temp_vmin, temp_vmax)
        rgba_temp = np.asarray(cmap(t01), dtype=float)  # (rings, 4)
        rgba_temp[:, 3] = layer_alpha

        # ring_patches are stored outer->inner; temp arrays are inner->outer.
        for patch_idx, p in enumerate(ring_patches):
            ring_idx = (rings - 1) - patch_idx
            p.set_facecolor(tuple(rgba_temp[ring_idx]))

        # Density -> particle spawn rate + particle base alpha.
        density01 = _normalize_01(density_rings)
        target_per_ring = np.rint(dots_min + (dots_max - dots_min) * density01).astype(int)
        target_per_ring = np.clip(target_per_ring, 0, dots_max)

        base_alpha_ring = float(args.alpha_min) + (float(args.alpha_max) - float(args.alpha_min)) * density01
        base_alpha_ring = np.clip(base_alpha_ring, 0.0, 1.0)

        rgba_particles = np.asarray(cmap(t01), dtype=float)
        rgba_particles[:, 3] = base_alpha_ring

        # Age existing particles and drop expired.
        nonlocal particle_offsets, particle_ring_idx, particle_age
        if particle_age.size:
            particle_age = particle_age + 1
            keep = particle_age < lifetime_frames
            particle_offsets = particle_offsets[keep]
            particle_ring_idx = particle_ring_idx[keep]
            particle_age = particle_age[keep]

        # Spawn new particles so the steady-state count approaches target_per_ring.
        if particle_ring_idx.size:
            current_counts = np.bincount(particle_ring_idx, minlength=rings)
        else:
            current_counts = np.zeros(rings, dtype=int)

        spawn_parts_offsets: list[np.ndarray] = []
        spawn_parts_ring_idx: list[np.ndarray] = []
        spawn_parts_age: list[np.ndarray] = []
        for ring_idx in range(rings):
            target = int(target_per_ring[ring_idx])

            # Spawn rate that yields ~target visible on average with lifetime L.
            desired_rate = float(target) / float(lifetime_frames)
            spawn_n = int(np.floor(desired_rate))
            if particle_rng.random() < (desired_rate - float(spawn_n)):
                spawn_n += 1

            # If we're far below target (e.g., sudden density spike), catch up gently.
            deficit = max(0, target - int(current_counts[ring_idx]))
            spawn_n = min(target, spawn_n + int(np.ceil(deficit * 0.15)))

            if spawn_n <= 0:
                continue

            bank = ring_point_banks[ring_idx]
            pick = particle_rng.integers(0, bank.shape[0], size=spawn_n, dtype=int)
            spawn_parts_offsets.append(bank[pick])
            spawn_parts_ring_idx.append(np.full((spawn_n,), ring_idx, dtype=int))
            spawn_parts_age.append(np.zeros((spawn_n,), dtype=int))

        if spawn_parts_offsets:
            particle_offsets = (
                np.concatenate([particle_offsets, *spawn_parts_offsets], axis=0)
                if particle_offsets.size
                else np.concatenate(spawn_parts_offsets, axis=0)
            )
            particle_ring_idx = (
                np.concatenate([particle_ring_idx, *spawn_parts_ring_idx], axis=0)
                if particle_ring_idx.size
                else np.concatenate(spawn_parts_ring_idx, axis=0)
            )
            particle_age = (
                np.concatenate([particle_age, *spawn_parts_age], axis=0)
                if particle_age.size
                else np.concatenate(spawn_parts_age, axis=0)
            )

        # Compute fade + slight shrink with age.
        if particle_age.size:
            age01 = np.clip(particle_age.astype(float) / float(lifetime_frames), 0.0, 1.0)
            fade = (1.0 - age01) ** 1.7
            sizes = float(args.dot_size) * (0.65 + 0.35 * fade)

            colors = rgba_particles[particle_ring_idx].copy()
            colors[:, 3] = colors[:, 3] * fade
        else:
            sizes = np.empty((0,), dtype=float)
            colors = np.empty((0, 4), dtype=float)

        dots.set_offsets(particle_offsets)
        dots.set_facecolors(colors)
        dots.set_sizes(sizes)

        return (*ring_patches, dots, time_text)

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1000 / args.fps,
        blit=True,
    )

    suffix = out_path.suffix.lower()
    savefig_kwargs = dict(facecolor="black", edgecolor="black")
    if suffix == ".mp4":
        ani.save(
            str(out_path),
            writer=animation.FFMpegWriter(fps=int(args.fps), bitrate=1800),
            savefig_kwargs=savefig_kwargs,
        )
    else:
        ani.save(str(out_path), writer=animation.PillowWriter(fps=int(args.fps)), savefig_kwargs=savefig_kwargs)

    plt.close(fig)
    print(f"Wrote axial animation: {out_path}")


if __name__ == "__main__":
    main()
