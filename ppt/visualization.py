
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class EscapeAnimationConfig:
	frames: int = 360
	fps: int = 30
	omega_rad_per_frame: float = math.tau / 120  # 1 revolution per 120 frames
	r_start: float = 2
	r_end: float = 1.25
	r_red: float = 1.0
	r_green_outer: float = 1.6
	dpi: int = 140
	figsize: tuple[float, float] = (6.2, 6.2)
	trail: bool = True
	trail_length: int = 90


def build_escape_animation(
	*,
	config: EscapeAnimationConfig,
	save_path: Optional[str | Path] = None,
	save_frames_dir: Optional[str | Path] = None,
	show: bool = True,
):
	save_path = Path(save_path) if save_path is not None else None
	save_frames_dir = Path(save_frames_dir) if save_frames_dir is not None else None

	import matplotlib

	if save_path is not None or save_frames_dir is not None:
		matplotlib.use("Agg", force=True)

	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.animation import FuncAnimation
	from matplotlib.patches import Circle, Wedge, Patch

	if config.r_green_outer <= config.r_red:
		raise ValueError("r_green_outer must be > r_red to form a ring")
	if config.frames <= 1:
		raise ValueError("frames must be > 1")
	if config.r_start <= 0:
		raise ValueError("r_start must be > 0 (do not start exactly at 0)")
	if config.r_end <= config.r_start:
		raise ValueError("r_end must be > r_start")
	if config.r_end <= 0:
		raise ValueError("r_end must be positive")

	fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
	ax.set_aspect("equal", adjustable="box")

	lim = max(config.r_green_outer, config.r_end) * 1.12
	ax.set_xlim(-lim, lim)
	ax.set_ylim(-lim, lim)

	red_disk = Circle(
		(0.0, 0.0),
		radius=config.r_red,
		facecolor="#ff6b6b",
		edgecolor="none",
		alpha=0.18,
		zorder=0,
	)
	ax.add_patch(red_disk)

	green_ring = Wedge(
		(0.0, 0.0),
		r=config.r_green_outer,
		theta1=0,
		theta2=360,
		width=config.r_green_outer - config.r_red,
		facecolor="#51cf66",
		edgecolor="none",
		alpha=0.18,
		zorder=0,
	)
	ax.add_patch(green_ring)

	red_boundary = Circle(
		(0.0, 0.0),
		radius=config.r_red,
		facecolor="none",
		edgecolor="#fa5252",
		linewidth=2.0,
		alpha=0.55,
		zorder=1,
	)
	ax.add_patch(red_boundary)

	outer_boundary = Circle(
		(0.0, 0.0),
		radius=config.r_green_outer,
		facecolor="none",
		edgecolor="#2f9e44",
		linewidth=1.5,
		alpha=0.35,
		zorder=1,
	)
	ax.add_patch(outer_boundary)

	legend_handles = [
		Patch(facecolor="#ff6b6b", edgecolor="none", alpha=0.25, label="Illness (red region)"),
		Patch(facecolor="#51cf66", edgecolor="none", alpha=0.25, label="Desired state (green ring)"),
	]
	ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.85)

	(point,) = ax.plot([], [], marker="o", markersize=8, linestyle="none", zorder=3)
	(trail_line,) = ax.plot([], [], linewidth=2.0, alpha=0.35, zorder=2)

	r = np.linspace(config.r_start, config.r_end, config.frames)
	theta = np.arange(config.frames, dtype=float) * config.omega_rad_per_frame
	x = r * np.cos(theta)
	y = r * np.sin(theta)

	def set_point_style(radius: float) -> None:
		if radius < config.r_red:
			point.set_markerfacecolor("#e03131")
			point.set_markeredgecolor("#e03131")
			point.set_alpha(0.95)
		else:
			point.set_markerfacecolor("#2b8a3e")
			point.set_markeredgecolor("#2b8a3e")
			point.set_alpha(0.95)

	def init():
		point.set_data([], [])
		trail_line.set_data([], [])
		set_point_style(r[0])
		return point, trail_line

	def update(frame_idx: int):
		point.set_data([x[frame_idx]], [y[frame_idx]])
		set_point_style(r[frame_idx])

		if config.trail:
			start = max(0, frame_idx - config.trail_length)
			trail_line.set_data(x[start : frame_idx + 1], y[start : frame_idx + 1])
			trail_line.set_color(point.get_markerfacecolor())
		else:
			trail_line.set_data([], [])

		return point, trail_line

	anim = FuncAnimation(
		fig,
		update,
		init_func=init,
		frames=config.frames,
		interval=1000 / config.fps,
		blit=True,
	)

	if save_frames_dir is not None:
		save_frames_dir.mkdir(parents=True, exist_ok=True)
		# Ensure artists exist at their initial state.
		init()
		# Render and save each frame as a PDF.
		for frame_idx in range(config.frames):
			update(frame_idx)
			fig.canvas.draw()
			out_path = save_frames_dir / f"anim_{frame_idx:04d}.pdf"
			fig.savefig(out_path, format="pdf", bbox_inches="tight")

	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		suffix = save_path.suffix.lower()

		if suffix == ".gif":
			anim.save(save_path, writer="pillow", fps=config.fps)
		elif suffix in {".mp4", ".m4v"}:
			anim.save(save_path, writer="ffmpeg", fps=config.fps)
		else:
			raise ValueError("Unsupported extension. Use .gif or .mp4")

	if show and save_path is None and save_frames_dir is None:
		plt.show()
	else:
		plt.close(fig)

	return anim


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description=(
			"Animate a rotating point whose radius increases until it exits a red "
			"disk and enters a green ring."
		)
	)
	p.add_argument(
		"--save",
		type=str,
		default="output/anim.gif",
		help="Output path (.gif or .mp4)",
	)
	p.add_argument(
		"--save-frames-dir",
		type=str,
		default="output/frames",
		help="If set, save per-frame PDFs to this directory (e.g. images/frames)",
	)
	p.add_argument("--no-show", action="store_true", help="Do not open a window")
	p.add_argument("--frames", type=int, default=360)
	p.add_argument("--fps", type=int, default=30)
	p.add_argument(
		"--omega",
		type=float,
		default=math.tau / 60,
		help="Angular speed (radians per frame)",
	)
	p.add_argument("--r-red", type=float, default=1.0)
	p.add_argument("--r-green-outer", type=float, default=1.6)
	p.add_argument("--r-start", type=float, default=0.05)
	p.add_argument("--r-end", type=float, default=1.25)
	p.add_argument("--dpi", type=int, default=140)
	p.add_argument("--no-trail", action="store_true")
	p.add_argument("--trail-length", type=int, default=90)
	return p.parse_args()


def main() -> None:
	args = _parse_args()
	cfg = EscapeAnimationConfig(
		frames=args.frames,
		fps=args.fps,
		omega_rad_per_frame=args.omega,
		r_start=args.r_start,
		r_end=args.r_end,
		r_red=args.r_red,
		r_green_outer=args.r_green_outer,
		dpi=args.dpi,
		trail=not args.no_trail,
		trail_length=args.trail_length,
	)
	build_escape_animation(
		config=cfg,
		save_path=args.save,
		save_frames_dir=args.save_frames_dir,
		show=not args.no_show,
	)


if __name__ == "__main__":
	main()
