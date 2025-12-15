"""2D top-down tokamak particle visualization (animation).

This produces a simple annulus (tokamak top-down view) and animates particles
moving toroidally. The number of particles in each radial ring is driven by a
time-dependent function (e.g. sinusoidal with per-ring phase shift).

Run as a script:
  python torax/visualization.py --out torax_outputs/particles.gif
"""

from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass

import numpy as np


def _require_script_execution() -> None:
	# Mirror the repo's execution convention: avoid module execution.
	if __package__:
		raise RuntimeError(
			"Run as a script: `python torax/visualization.py` (not `python -m torax.visualization`)."
		)


@dataclass(frozen=True)
class DensitySpec:
	mode: str
	base_profile: str
	amp: float
	period: float
	min_frac: float


def _base_profile_weights(*, rings: int, base_profile: str) -> np.ndarray:
	# Returns positive weights (shape: [rings]).
	if rings <= 0:
		raise ValueError("rings must be positive")
	idx = np.arange(rings, dtype=float)
	# Normalize radius proxy in [0, 1].
	x = idx / max(1.0, rings - 1)

	if base_profile == "flat":
		w = np.ones(rings, dtype=float)
	elif base_profile == "peaked":
		# More particles near the core (small x).
		w = 1.0 - 0.75 * x
	elif base_profile == "hollow":
		# More particles near the edge.
		w = 0.25 + 0.75 * x
	else:
		raise ValueError(f"Unknown base_profile={base_profile!r}")

	# Ensure strictly positive.
	w = np.clip(w, 1e-6, None)
	return w


def _density_per_ring(*, t: float, rings: int, spec: DensitySpec) -> np.ndarray:
	"""Returns non-negative density weights per ring at time t."""
	base = _base_profile_weights(rings=rings, base_profile=spec.base_profile)

	if spec.mode == "static":
		w = base
	elif spec.mode == "sin":
		# Per-ring phase shift makes a visible rotating modulation.
		phase = 2.0 * math.pi * (np.arange(rings, dtype=float) / max(1.0, rings))
		osc = np.sin(2.0 * math.pi * (t / max(1e-9, spec.period)) + phase)
		mod = 1.0 + spec.amp * osc
		# Keep a minimum fraction of the base to avoid rings emptying completely.
		mod = np.clip(mod, spec.min_frac, None)
		w = base * mod
	elif spec.mode == "pulse":
		# A Gaussian pulse sweeping outward over time.
		center = (t / max(1e-9, spec.period)) % 1.0
		x = np.arange(rings, dtype=float) / max(1.0, rings - 1)
		sigma = 0.12
		pulse = np.exp(-0.5 * ((x - center) / sigma) ** 2)
		w = base * (spec.min_frac + spec.amp * pulse)
	else:
		raise ValueError(f"Unknown density mode={spec.mode!r}")

	return np.clip(w, 0.0, None)


def _counts_from_density(*, weights: np.ndarray, n_total: int) -> np.ndarray:
	"""Converts weights into integer per-ring particle counts that sum to n_total."""
	if n_total < 0:
		raise ValueError("n_total must be >= 0")
	if weights.ndim != 1:
		raise ValueError("weights must be 1D")
	if weights.size == 0:
		return np.zeros(0, dtype=int)
	wsum = float(np.sum(weights))
	if not math.isfinite(wsum) or wsum <= 0.0:
		# Degenerate: put everything in ring 0.
		out = np.zeros(weights.size, dtype=int)
		if weights.size:
			out[0] = n_total
		return out

	ideal = (weights / wsum) * float(n_total)
	counts = np.floor(ideal).astype(int)
	remaining = int(n_total - int(np.sum(counts)))
	if remaining > 0:
		# Distribute remainder to the largest fractional parts.
		frac = ideal - counts
		order = np.argsort(-frac)
		counts[order[:remaining]] += 1
	return counts


def _sample_r_in_ring(
	*,
	rng: np.random.Generator,
	r_inner: float,
	r_outer: float,
	ring_index: int,
	rings: int,
	n: int,
) -> np.ndarray:
	# Sample uniformly in annulus area within this ring.
	if rings <= 0:
		raise ValueError("rings must be positive")
	dr = (r_outer - r_inner) / float(rings)
	r0 = r_inner + ring_index * dr
	r1 = r0 + dr
	# Uniform in area => r^2 uniform.
	u = rng.random(n)
	r = np.sqrt((r0 * r0) + u * ((r1 * r1) - (r0 * r0)))
	return r


def _ensure_annulus_axes(*, ax, r_inner: float, r_outer: float) -> None:
	# Draw tokamak top-down annulus.
	import matplotlib.patches as patches

	ax.set_aspect("equal", adjustable="box")
	ax.set_xlim(-r_outer * 1.05, r_outer * 1.05)
	ax.set_ylim(-r_outer * 1.05, r_outer * 1.05)
	ax.set_xticks([])
	ax.set_yticks([])

	outer = patches.Circle((0.0, 0.0), radius=r_outer, fill=False, linewidth=1.5)
	inner = patches.Circle((0.0, 0.0), radius=r_inner, fill=False, linewidth=1.5)
	ax.add_patch(outer)
	ax.add_patch(inner)


def animate_particles(
	*,
	out_path: pathlib.Path,
	r_inner: float,
	r_outer: float,
	rings: int,
	seconds: float,
	fps: int,
	n_total: int,
	seed: int,
	density_spec: DensitySpec,
	trail_len: int,
	n_tracers: int,
	tracer_trail_len: int,
	omega: float,
	omega_shear: float,
	r_jitter: float,
	show: bool,
) -> pathlib.Path:
	"""Create and save an animation, returning the written path."""
	if r_outer <= r_inner:
		raise ValueError("Expected r_outer > r_inner")
	if rings <= 0:
		raise ValueError("rings must be positive")
	if fps <= 0:
		raise ValueError("fps must be positive")
	if seconds <= 0:
		raise ValueError("seconds must be positive")
	if trail_len < 0:
		raise ValueError("trail_len must be >= 0")
	if n_tracers < 0:
		raise ValueError("n_tracers must be >= 0")
	if tracer_trail_len < 0:
		raise ValueError("tracer_trail_len must be >= 0")

	# Headless-safe plotting.
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.animation as animation
	import matplotlib.pyplot as plt

	rng = np.random.default_rng(seed)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	frames = int(math.ceil(seconds * fps))
	dt = 1.0 / float(fps)

	# Particle state arrays (variable length): r, theta, ring_index
	r_vals = np.zeros(0, dtype=float)
	theta_vals = np.zeros(0, dtype=float)
	ring_idx = np.zeros(0, dtype=int)

	# Dedicated tracer particles (fixed count) so motion is visible even when the
	# bulk distribution is axisymmetric.
	tr_r = np.zeros(0, dtype=float)
	tr_theta = np.zeros(0, dtype=float)
	tr_ring = np.zeros(0, dtype=int)

	# Trail buffer: previous positions (all particles), newest last.
	trail: list[np.ndarray] = []
	tracer_trail: list[np.ndarray] = []

	fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
	_ensure_annulus_axes(ax=ax, r_inner=r_inner, r_outer=r_outer)

	# We draw the cloud trail as multiple scatters with increasing alpha.
	trail_scatters = []
	for _ in range(max(0, trail_len)):
		sc = ax.scatter([], [], s=4, c="#1f77b4", alpha=0.0, linewidths=0)
		trail_scatters.append(sc)
	particles_scatter = ax.scatter([], [], s=8, c="#1f77b4", alpha=0.95, linewidths=0)

	# Tracer artists (lines + points).
	tracer_lines = []
	for _ in range(max(0, n_tracers)):
		(line,) = ax.plot([], [], color="#ff7f0e", linewidth=1.0, alpha=0.6)
		tracer_lines.append(line)
	tracer_scatter = ax.scatter([], [], s=26, c="#ff7f0e", alpha=0.95, linewidths=0)
	text = ax.text(
		0.02,
		0.98,
		"",
		transform=ax.transAxes,
		va="top",
		ha="left",
		fontsize=9,
	)

	def _resample_to_counts(target_counts: np.ndarray) -> None:
		nonlocal r_vals, theta_vals, ring_idx
		# Adjust particles ring-by-ring to match target_counts.
		new_r: list[np.ndarray] = []
		new_theta: list[np.ndarray] = []
		new_ring: list[np.ndarray] = []
		dr = (r_outer - r_inner) / float(rings)

		for i in range(rings):
			target = int(target_counts[i])
			mask = ring_idx == i
			existing_r = r_vals[mask]
			existing_theta = theta_vals[mask]
			existing_n = int(existing_r.size)

			if existing_n >= target:
				# Keep a random subset to avoid bias.
				if target > 0:
					keep = rng.choice(existing_n, size=target, replace=False)
					new_r.append(existing_r[keep])
					new_theta.append(existing_theta[keep])
					new_ring.append(np.full(target, i, dtype=int))
				continue

			# Keep all existing and add more.
			if existing_n > 0:
				new_r.append(existing_r)
				new_theta.append(existing_theta)
				new_ring.append(np.full(existing_n, i, dtype=int))
			add = target - existing_n
			if add > 0:
				add_r = _sample_r_in_ring(
					rng=rng,
					r_inner=r_inner,
					r_outer=r_outer,
					ring_index=i,
					rings=rings,
					n=add,
				)
				add_theta = rng.random(add) * (2.0 * math.pi)
				# Slightly concentrate new particles toward their ring center.
				mid = r_inner + (i + 0.5) * dr
				add_r = np.clip(
					add_r + rng.normal(0.0, 0.15 * dr, size=add),
					mid - 0.5 * dr,
					mid + 0.5 * dr,
				)
				new_r.append(add_r)
				new_theta.append(add_theta)
				new_ring.append(np.full(add, i, dtype=int))

		if new_r:
			r_vals = np.concatenate(new_r)
			theta_vals = np.concatenate(new_theta)
			ring_idx = np.concatenate(new_ring)
		else:
			r_vals = np.zeros(0, dtype=float)
			theta_vals = np.zeros(0, dtype=float)
			ring_idx = np.zeros(0, dtype=int)

	def _step_state() -> None:
		nonlocal r_vals, theta_vals
		if r_vals.size == 0:
			return
		# Toroidal angular velocity with optional shear by ring index.
		ring_frac = ring_idx.astype(float) / max(1.0, rings - 1)
		w = omega * (1.0 + omega_shear * (ring_frac - 0.5))
		theta_vals = (
			theta_vals + w * dt + rng.normal(0.0, 0.05, size=r_vals.size) * dt
		) % (2.0 * math.pi)

		if r_jitter > 0.0:
			dr = (r_outer - r_inner) / float(rings)
			r0 = r_inner + ring_idx * dr
			r1 = r0 + dr
			r_vals = np.clip(
				r_vals + rng.normal(0.0, r_jitter * dr, size=r_vals.size),
				r0 + 1e-6,
				r1 - 1e-6,
			)

	def _step_tracers() -> None:
		nonlocal tr_r, tr_theta
		if tr_r.size == 0:
			return
		ring_frac = tr_ring.astype(float) / max(1.0, rings - 1)
		w = omega * (1.0 + omega_shear * (ring_frac - 0.5))
		tr_theta = (
			tr_theta + w * dt + rng.normal(0.0, 0.02, size=tr_r.size) * dt
		) % (2.0 * math.pi)

		if r_jitter > 0.0:
			dr = (r_outer - r_inner) / float(rings)
			r0 = r_inner + tr_ring * dr
			r1 = r0 + dr
			tr_r = np.clip(
				tr_r + rng.normal(0.0, 0.35 * r_jitter * dr, size=tr_r.size),
				r0 + 1e-6,
				r1 - 1e-6,
			)

	def _positions_xy() -> np.ndarray:
		if r_vals.size == 0:
			return np.zeros((0, 2), dtype=float)
		x = r_vals * np.cos(theta_vals)
		y = r_vals * np.sin(theta_vals)
		return np.column_stack([x, y])

	def _tracer_positions_xy() -> np.ndarray:
		if tr_r.size == 0:
			return np.zeros((0, 2), dtype=float)
		x = tr_r * np.cos(tr_theta)
		y = tr_r * np.sin(tr_theta)
		return np.column_stack([x, y])

	def _init_tracers() -> None:
		nonlocal tr_r, tr_theta, tr_ring
		if n_tracers <= 0:
			tr_r = np.zeros(0, dtype=float)
			tr_theta = np.zeros(0, dtype=float)
			tr_ring = np.zeros(0, dtype=int)
			return
		# Spread tracers across rings so shear is visible too.
		tr_ring = rng.integers(0, rings, size=n_tracers, dtype=int)
		tr_r_parts = []
		for i in range(rings):
			count = int(np.sum(tr_ring == i))
			if count <= 0:
				continue
			tr_r_parts.append(
				_sample_r_in_ring(
					rng=rng,
					r_inner=r_inner,
					r_outer=r_outer,
					ring_index=i,
					rings=rings,
					n=count,
				)
			)
		if tr_r_parts:
			tr_r = np.concatenate(tr_r_parts)
			# Keep tr_r aligned with tr_ring ordering (by reconstructing in that order).
			# We do it by re-sampling per tracer ring in the original tr_ring sequence.
			offset = 0
			tr_r_ordered = np.empty(n_tracers, dtype=float)
			for i in range(rings):
				count = int(np.sum(tr_ring == i))
				if count <= 0:
					continue
				mask = tr_ring == i
				tr_r_ordered[mask] = tr_r[offset : offset + count]
				offset += count
			tr_r = tr_r_ordered
		else:
			tr_r = np.zeros(0, dtype=float)
		tr_theta = rng.random(n_tracers) * (2.0 * math.pi)

	def init():
		_init_tracers()
		particles_scatter.set_offsets(np.zeros((0, 2)))
		for sc in trail_scatters:
			sc.set_offsets(np.zeros((0, 2)))
		for ln in tracer_lines:
			ln.set_data([], [])
		tracer_scatter.set_offsets(np.zeros((0, 2)))
		text.set_text("")
		return [particles_scatter, tracer_scatter, text, *trail_scatters, *tracer_lines]

	def update(frame_idx: int):
		# Update ring particle counts from density(t).
		t = frame_idx * dt
		weights = _density_per_ring(t=t, rings=rings, spec=density_spec)
		target_counts = _counts_from_density(weights=weights, n_total=n_total)
		_resample_to_counts(target_counts)

		# Advance particle motion.
		_step_state()
		_step_tracers()
		xy = _positions_xy()
		tr_xy = _tracer_positions_xy()

		# Update trail buffer.
		if trail_len > 0:
			trail.append(xy.copy())
			if len(trail) > trail_len:
				trail.pop(0)
		if tracer_trail_len > 0 and tr_xy.size:
			tracer_trail.append(tr_xy.copy())
			if len(tracer_trail) > tracer_trail_len:
				tracer_trail.pop(0)

		# Draw trail (newest strongest, older fainter).
		for k, sc in enumerate(trail_scatters):
			if k < len(trail):
				idx = len(trail) - 1 - k
				alpha = 0.08 + 0.65 * (1.0 - (k / max(1.0, trail_len - 1)))
				sc.set_offsets(trail[idx])
				sc.set_alpha(alpha)
			else:
				sc.set_offsets(np.zeros((0, 2)))
				sc.set_alpha(0.0)

		particles_scatter.set_offsets(xy)
		tracer_scatter.set_offsets(tr_xy)
		if tracer_lines:
			if tracer_trail_len <= 0 or not tracer_trail:
				for ln in tracer_lines:
					ln.set_data([], [])
			else:
				hist = np.stack(tracer_trail, axis=0)  # [T, n_tracers, 2]
				for i, ln in enumerate(tracer_lines):
					if i >= hist.shape[1]:
						ln.set_data([], [])
						continue
					ln.set_data(hist[:, i, 0], hist[:, i, 1])
		text.set_text(
			f"t={t:5.2f}s  N={xy.shape[0]}  rings={rings}  mode={density_spec.mode}  tracers={tr_xy.shape[0]}"
		)
		return [particles_scatter, tracer_scatter, text, *trail_scatters, *tracer_lines]

	ani = animation.FuncAnimation(
		fig,
		update,
		init_func=init,
		frames=frames,
		interval=1000.0 / float(fps),
		blit=True,
	)

	# Save.
	suffix = out_path.suffix.lower()
	if suffix not in {".gif", ".mp4"}:
		raise ValueError("--out must end with .gif or .mp4")

	if suffix == ".mp4":
		writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
		ani.save(str(out_path), writer=writer)
	else:
		# PillowWriter requires pillow installed.
		writer = animation.PillowWriter(fps=fps)
		ani.save(str(out_path), writer=writer)

	if show:
		# Note: we used Agg for headless safety; show may no-op on some setups.
		plt.show()
	plt.close(fig)
	return out_path


def main() -> None:
	_require_script_execution()

	parser = argparse.ArgumentParser(description="Tokamak top-down particle animation")
	parser.add_argument(
		"--out",
		type=pathlib.Path,
		default=pathlib.Path("torax_outputs/particles.gif"),
		help="Output animation path (.gif or .mp4)",
	)
	parser.add_argument("--rings", type=int, default=12, help="Number of radial rings")
	parser.add_argument(
		"--r-inner", type=float, default=0.65, help="Inner radius of tokamak annulus"
	)
	parser.add_argument(
		"--r-outer", type=float, default=1.0, help="Outer radius of tokamak annulus"
	)
	parser.add_argument("--seconds", type=float, default=8.0, help="Animation duration")
	parser.add_argument("--fps", type=int, default=30, help="Frames per second")
	parser.add_argument(
		"--n-total",
		type=int,
		default=1400,
		help="Total particles (distributed across rings)",
	)
	parser.add_argument("--seed", type=int, default=7, help="RNG seed")

	parser.add_argument(
		"--density-mode",
		choices=["static", "sin", "pulse"],
		default="sin",
		help="How per-ring particle density varies in time",
	)
	parser.add_argument(
		"--base-profile",
		choices=["flat", "peaked", "hollow"],
		default="peaked",
		help="Baseline radial distribution (before time modulation)",
	)
	parser.add_argument("--amp", type=float, default=0.65, help="Density modulation amplitude")
	parser.add_argument("--period", type=float, default=3.0, help="Density modulation period [s]")
	parser.add_argument(
		"--min-frac",
		type=float,
		default=0.15,
		help="Minimum fraction of baseline density to keep (avoid empty rings)",
	)

	parser.add_argument("--trail-len", type=int, default=14, help="Fading trail length (frames)")
	parser.add_argument(
		"--n-tracers",
		type=int,
		default=20,
		help="Number of highlighted tracer particles (makes rotation visible)",
	)
	parser.add_argument(
		"--tracer-trail-len",
		type=int,
		default=20,
		help="Tracer trail length (frames); 0 disables tracer trails",
	)
	parser.add_argument("--omega", type=float, default=6.0, help="Base angular velocity [rad/s]")
	parser.add_argument(
		"--omega-shear",
		type=float,
		default=0.35,
		help="Angular velocity shear vs radius (0=no shear)",
	)
	parser.add_argument(
		"--r-jitter",
		type=float,
		default=0.06,
		help="Radial random-walk jitter as fraction of ring width",
	)
	parser.add_argument("--show", action="store_true", help="Try to show an interactive window")

	args = parser.parse_args()

	density_spec = DensitySpec(
		mode=args.density_mode,
		base_profile=args.base_profile,
		amp=float(args.amp),
		period=float(args.period),
		min_frac=float(args.min_frac),
	)

	written = animate_particles(
		out_path=args.out,
		r_inner=float(args.r_inner),
		r_outer=float(args.r_outer),
		rings=int(args.rings),
		seconds=float(args.seconds),
		fps=int(args.fps),
		n_total=int(args.n_total),
		seed=int(args.seed),
		density_spec=density_spec,
		trail_len=int(args.trail_len),
		n_tracers=int(args.n_tracers),
		tracer_trail_len=int(args.tracer_trail_len),
		omega=float(args.omega),
		omega_shear=float(args.omega_shear),
		r_jitter=float(args.r_jitter),
		show=bool(args.show),
	)
	print(f"Wrote animation: {written}")


if __name__ == "__main__":
	main()
