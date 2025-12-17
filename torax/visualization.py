import argparse
import pathlib
import matplotlib
import xarray as xr
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import numpy as np
from collections import deque


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


def counts_from_weights(weights: np.ndarray, n_total: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weights must be 1D")
    if n_total < 0:
        raise ValueError("--n-total must be >= 0")
    s = float(np.sum(weights))
    if not np.isfinite(s) or s <= 0.0:
        out = np.zeros(weights.size, dtype=int)
        if out.size:
            out[0] = int(n_total)
        return out
    ideal = (weights / s) * float(n_total)
    out = np.floor(ideal).astype(int)
    rem = int(n_total - int(np.sum(out)))
    if rem > 0:
        frac = ideal - out
        out[np.argsort(-frac)[:rem]] += 1
    return out


def sample_points(rng: np.random.Generator, r_inner: float, r_outer: float, counts: np.ndarray) -> np.ndarray:
    """Sample points in an annulus with per-ring counts; returns xy array [N,2]."""
    rings = int(counts.size)
    dr = (float(r_outer) - float(r_inner)) / float(rings)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i, n in enumerate(counts.astype(int)):
        if n <= 0:
            continue
        r0 = float(r_inner) + float(i) * dr
        r1 = r0 + dr
        u = rng.random(n)
        r = np.sqrt((r0 * r0) + u * ((r1 * r1) - (r0 * r0)))
        theta = rng.random(n) * (2.0 * np.pi)
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))
    if not xs:
        return np.zeros((0, 2), dtype=float)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return np.column_stack([x, y])


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal tokamak particle animation from iter.nc")
    parser.add_argument("--iter-nc", type=pathlib.Path, default=pathlib.Path("torax/iter.nc"))
    parser.add_argument(
        "--iter-var",
        choices=["n_e", "n_i"],
        default="n_e",
        help="Profile used for particle density/alpha",
    )
    parser.add_argument("--temp-var", type=str, default="T_e", help="Profile used for particle colormap")
    parser.add_argument("--cmap", type=str, default="inferno", help="Matplotlib colormap for temperature")
    parser.add_argument("--temp-vmin", type=float, default=None, help="Colorbar/colormap min for temperature")
    parser.add_argument("--temp-vmax", type=float, default=None, help="Colorbar/colormap max for temperature")
    parser.add_argument("--alpha-min", type=float, default=0.10)
    parser.add_argument("--alpha-max", type=float, default=0.95)
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument(
        "--trail-frames",
        type=int,
        default=20,
        help="How many previous frames to keep as a fading particle trail (0 disables).",
    )
    parser.add_argument(
        "--trail-alpha",
        type=float,
        default=0.35,
        help="Multiplier on per-particle alpha for the trail (0 disables).",
    )
    parser.add_argument("--trail-linewidth", type=float, default=1.0)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("torax/particles.gif"))
    parser.add_argument("--rings", type=int, default=12)
    parser.add_argument("--r-inner", type=float, default=0.65)
    parser.add_argument("--r-outer", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n-total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
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

    # Pull underlying arrays once; we will interpolate manually for smooth animation.
    density_arr = np.asarray(da_density.values, dtype=float)
    temp_arr = np.asarray(da_temp.values, dtype=float)

    if args.temp_vmin is None:
        temp_vmin = float(np.nanmin(np.asarray(da_temp.values, dtype=float)))
    else:
        temp_vmin = float(args.temp_vmin)
    if args.temp_vmax is None:
        temp_vmax = float(np.nanmax(np.asarray(da_temp.values, dtype=float)))
    else:
        temp_vmax = float(args.temp_vmax)
    if not np.isfinite(temp_vmin) or not np.isfinite(temp_vmax) or temp_vmax <= temp_vmin:
        temp_vmin, temp_vmax = 0.0, 1.0

    out_path: pathlib.Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(np.ceil(float(args.seconds) * float(args.fps)))
    rng = np.random.default_rng(int(args.seed))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=0.86, bottom=0, top=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-args.r_outer * 1.05, args.r_outer * 1.05)
    ax.set_ylim(-args.r_outer * 1.05, args.r_outer * 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(
        patches.Circle((0.0, 0.0), radius=args.r_outer, fill=False, linewidth=1.5, edgecolor="white")
    )
    ax.add_patch(
        patches.Circle((0.0, 0.0), radius=args.r_inner, fill=False, linewidth=1.5, edgecolor="white")
    )

    cmap = matplotlib.colormaps.get_cmap(str(args.cmap))
    norm = Normalize(vmin=temp_vmin, vmax=temp_vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(str(args.temp_var), color="white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")
    trail_frames = max(0, int(args.trail_frames))
    trail_alpha = float(args.trail_alpha)

    trail_lc: LineCollection | None = None
    # (xy, rgba, respawned_mask)
    trail_history: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    if trail_frames > 0 and trail_alpha > 0.0:
        trail_lc = LineCollection([], linewidths=float(args.trail_linewidth), zorder=1)
        ax.add_collection(trail_lc)
        trail_history = deque(maxlen=trail_frames + 1)  # +1 so we can drop current frame from trail

    scat = ax.scatter([], [], s=float(args.point_size), linewidths=0)

    rings = int(args.rings)
    r_inner = float(args.r_inner)
    r_outer = float(args.r_outer)
    dr = (r_outer - r_inner) / float(rings)
    n_total = int(args.n_total)

    # Persistent particle state: smooth motion comes from keeping identities/positions.
    particle_r = np.zeros(n_total, dtype=float)
    particle_theta = rng.random(n_total) * (2.0 * np.pi)
    particle_ring = np.zeros(n_total, dtype=int)
    particle_omega = np.zeros(n_total, dtype=float)

    # Simple smooth angular motion (per-ring). Higher rings rotate a bit faster.
    # (No new CLI args; adjust here if you want different dynamics.)
    omega_ring = np.linspace(0.8, 2.4, rings, dtype=float)  # rad/s

    def _sample_r_in_ring(ring_index: int, n: int) -> np.ndarray:
        r0 = r_inner + float(ring_index) * dr
        r1 = r0 + dr
        u = rng.random(n)
        return np.sqrt((r0 * r0) + u * ((r1 * r1) - (r0 * r0)))

    def _assign_particles_to_target_counts(target_counts: np.ndarray) -> np.ndarray:
        """Reassign particle_ring (and respawn r/theta) to match target_counts.

        Returns a boolean mask indicating which particles were respawned. This is used
        to break trail segments so we don't draw long "teleport" lines.
        """
        nonlocal particle_r, particle_theta, particle_ring, particle_omega
        target_counts = np.asarray(target_counts, dtype=int)
        if target_counts.size != rings:
            raise ValueError("target_counts must have length == rings")
        if int(np.sum(target_counts)) != n_total:
            # Be forgiving; normalize by adjusting ring 0.
            target_counts = target_counts.copy()
            target_counts[0] += int(n_total - int(np.sum(target_counts)))

        respawned = np.zeros(n_total, dtype=bool)

        # Current membership lists.
        members = [np.where(particle_ring == i)[0] for i in range(rings)]
        free: list[int] = []

        # Collect excess particles into a free pool.
        for i in range(rings):
            cur = int(members[i].size)
            want = int(target_counts[i])
            if cur > want:
                excess = members[i][want:]
                free.extend(excess.tolist())

        # Fill deficits from the free pool.
        free_pos = 0
        for i in range(rings):
            cur = int(members[i].size)
            want = int(target_counts[i])
            deficit = want - cur
            if deficit <= 0:
                continue
            take = free[free_pos : free_pos + deficit]
            free_pos += deficit
            if not take:
                continue

            idx = np.asarray(take, dtype=int)
            particle_ring[idx] = i
            particle_r[idx] = _sample_r_in_ring(i, idx.size)
            particle_theta[idx] = rng.random(idx.size) * (2.0 * np.pi)
            # Per-particle omega variation keeps motion from looking rigid.
            particle_omega[idx] = omega_ring[i] * (0.85 + 0.30 * rng.random(idx.size))

            respawned[idx] = True

        return respawned

    def _interp_profiles(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate (density, temp) profiles across stored time samples."""
        t_len = int(density_arr.shape[0])
        if t_len <= 1:
            return density_arr[0], temp_arr[0]
        # Map animation frames to fractional index across available timesteps.
        denom = max(1, int(frames) - 1)
        pos = (float(frame_index % frames) / float(denom)) * float(t_len - 1)
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, t_len - 1)
        w = float(pos - float(i0))
        prof_d = (1.0 - w) * density_arr[i0] + w * density_arr[i1]
        prof_t = (1.0 - w) * temp_arr[i0] + w * temp_arr[i1]
        return prof_d, prof_t

    # Initialize particle rings based on the first profile.
    init_density = density_arr[0]
    init_weights = ring_weights_from_profile(rho, np.asarray(init_density, dtype=float), rings)
    init_counts = counts_from_weights(init_weights, n_total)
    # Seed particle_ring in a deterministic block layout, then materialize r/theta.
    start = 0
    for i in range(rings):
        n_i = int(init_counts[i])
        if n_i <= 0:
            continue
        sl = slice(start, start + n_i)
        particle_ring[sl] = i
        particle_r[sl] = _sample_r_in_ring(i, n_i)
        particle_omega[sl] = omega_ring[i] * (0.85 + 0.30 * rng.random(n_i))
        start += n_i

    def init():
        if trail_lc is not None:
            trail_lc.set_segments([])
            trail_lc.set_color(np.zeros((0, 4), dtype=float))
        if trail_history is not None:
            trail_history.clear()
        scat.set_offsets(np.zeros((0, 2)))
        scat.set_facecolors(np.zeros((0, 4), dtype=float))
        return (trail_lc, scat) if trail_lc is not None else (scat,)

    def update(frame: int):
        prof_density, prof_temp = _interp_profiles(int(frame))

        density_rings = ring_weights_from_profile(rho, np.asarray(prof_density, dtype=float), rings)
        temp_rings = ring_weights_from_profile(rho, np.asarray(prof_temp, dtype=float), rings)

        target_counts = counts_from_weights(density_rings, n_total)
        respawned = _assign_particles_to_target_counts(target_counts)

        # Integrate a simple angular motion for smooth trajectories.
        dt_anim = 1.0 / float(args.fps)
        particle_theta[:] = (particle_theta + particle_omega * dt_anim) % (2.0 * np.pi)
        xy = np.column_stack([particle_r * np.cos(particle_theta), particle_r * np.sin(particle_theta)])
        scat.set_offsets(xy)

        if xy.shape[0] == 0:
            scat.set_facecolors(np.zeros((0, 4), dtype=float))
            if trail_lc is not None:
                trail_lc.set_segments([])
                trail_lc.set_color(np.zeros((0, 4), dtype=float))
            return (trail_lc, scat) if trail_lc is not None else (scat,)

        ring_idx = particle_ring

        alpha01 = _normalize_01(density_rings)
        alpha_ring = float(args.alpha_min) + (float(args.alpha_max) - float(args.alpha_min)) * alpha01
        alpha = alpha_ring[ring_idx]

        t01 = _normalize_range_01(temp_rings, temp_vmin, temp_vmax)
        rgba_ring = cmap(t01)
        rgba = rgba_ring[ring_idx].copy()
        rgba[:, 3] = np.clip(alpha, 0.0, 1.0)

        if trail_lc is not None and trail_history is not None:
            trail_history.append((xy.copy(), rgba.copy(), respawned.copy()))
            hist = list(trail_history)
            if len(hist) <= 1:
                trail_lc.set_segments([])
                trail_lc.set_color(np.zeros((0, 4), dtype=float))
            else:
                # Build line segments between consecutive history frames for each particle index.
                segments_list: list[np.ndarray] = []
                colors_list: list[np.ndarray] = []
                seg_count = len(hist) - 1
                fade = (
                    np.linspace(0.0, 1.0, seg_count, endpoint=True)
                    if seg_count > 1
                    else np.array([1.0], dtype=float)
                )
                for k in range(seg_count):
                    xy0, _, _ = hist[k]
                    xy1, rgba1, resp1 = hist[k + 1]
                    if xy0.shape != xy1.shape or xy0.size == 0:
                        continue

                    # Break trails for particles that were respawned at this step.
                    connect_mask = ~np.asarray(resp1, dtype=bool)
                    if not np.any(connect_mask):
                        continue
                    idx = np.where(connect_mask)[0]

                    segs = np.stack([xy0[idx], xy1[idx]], axis=1)  # [M, 2, 2]
                    c = rgba1[idx].copy()
                    c[:, 3] = np.clip(c[:, 3] * (trail_alpha * float(fade[k])), 0.0, 1.0)
                    segments_list.append(segs)
                    colors_list.append(c)

                if segments_list:
                    segments = np.concatenate(segments_list, axis=0)
                    colors = np.concatenate(colors_list, axis=0)
                    trail_lc.set_segments(segments)
                    trail_lc.set_color(colors)
                else:
                    trail_lc.set_segments([])
                    trail_lc.set_color(np.zeros((0, 4), dtype=float))

        scat.set_facecolors(rgba)
        return (trail_lc, scat) if trail_lc is not None else (scat,)

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=1000 / args.fps, blit=True)

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
    print(f"Wrote animation: {out_path}")


if __name__ == "__main__":
    main()
