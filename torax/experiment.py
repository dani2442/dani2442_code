"""TORAX demo: run a tiny transport simulation + plot results.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

import numpy as np


def _require_script_execution() -> None:
	# When run as `python -m torax.experiment`, __package__ will be non-empty.
	# That execution mode is incompatible with this repo layout (see docstring).
	if __package__:
		raise RuntimeError(
			"Run as a script: `python torax/experiment.py` (not `python -m torax.experiment`)."
		)


def _import_torax_lib():
	"""Imports the installed `torax` library safely.

	We delay this import so we can reject module execution first.
	"""

	import torax  # type: ignore

	return torax


def _build_base_config_dict() -> dict[str, Any]:
	"""Returns a small-but-explicit config dict suitable for a quick demo.

	The upstream project provides many example configs. The simplest is
	`torax.examples.basic_config`, which uses a circular geometry and lots of
	defaults.

	In this demo we start from that config, but also set a few key parameters
	explicitly so the run is:
	  - fast (small t_final)
	  - likely to produce non-trivial profiles (enable evolution)
	  - stable (reasonable initial/boundary values)
	"""

	torax = _import_torax_lib()

	try:
		from torax.examples import basic_config  # type: ignore

		base = dict(basic_config.CONFIG)
	except Exception:
		# Fallback: minimal config if examples aren't packaged (uncommon).
		base = {
			"profile_conditions": {},
			"plasma_composition": {},
			"numerics": {},
			"geometry": {"geometry_type": "circular"},
			"neoclassical": {"bootstrap_current": {}},
		}

	# Make the simulation short, for a fast demo.
	# (If you want a more realistic run, increase t_final and/or reduce fixed_dt.)
	base["numerics"] = {
		**base.get("numerics", {}),
		"t_final": 0.25,
		"fixed_dt": 0.025,
		# Make sure the physics is actually evolved.
		"evolve_ion_heat": True,
		"evolve_electron_heat": True,
		"evolve_current": True,
		"evolve_density": True,
		# Keep things conservative.
		"adaptive_dt": False,
	}

	# Provide a few explicit initial/boundary conditions.
	# Units follow TORAX conventions (see docs): temperatures in keV, densities in m^-3.
	base["profile_conditions"] = {
		**base.get("profile_conditions", {}),
		"Ip": 3.0e6,  # Plasma current [A]
		"n_e": 1.0e20,
		"n_e_right_bc": 1.0e20,
		"n_e_nbar_is_fGW": False,
		"T_i": 6.0,
		"T_e": 6.0,
		"T_i_right_bc": 0.2,
		"T_e_right_bc": 0.2,
	}

	# Ensure circular geometry if the example dict omitted it.
	base["geometry"] = {**base.get("geometry", {}), "geometry_type": "circular"}
	return base


def _run_case(
	*,
	name: str,
	config_dict: dict[str, Any],
	out_dir: pathlib.Path,
	log_timestep_info: bool,
	progress_bar: bool,
):
	torax = _import_torax_lib()

	torax.log_jax_backend()
	torax.set_jax_precision()

	torax_config = torax.ToraxConfig.from_dict(config_dict)
	data_tree, state_history = torax.run_simulation(
		torax_config,
		log_timestep_info=log_timestep_info,
		progress_bar=progress_bar,
	)
	if state_history.sim_error != torax.SimError.NO_ERROR:
		raise RuntimeError(f"Simulation '{name}' ended with error: {state_history.sim_error}")

	# Save the output using TORAX's helper (the official outputs are NetCDF files).
	# This is a convenient bridge to the `plot_torax` tool later, too.
	from torax._src import simulation_app as simulation_app_lib  # type: ignore

	out_dir.mkdir(parents=True, exist_ok=True)
	outfile = out_dir / f"{name}.nc"
	simulation_app_lib.write_output_to_file(str(outfile), data_tree)

	return data_tree, state_history, outfile


def _radial_coord(arr) -> np.ndarray:
	"""Returns the most likely radial coordinate values for a profile array."""
	# Profile arrays are typically (time, rho_cell_norm) or (time, rho_face_norm).
	# We'll just detect the radial dimension name.
	dims = list(arr.dims)
	radial_dim = None
	for d in dims:
		if "rho" in d and d != "time":
			radial_dim = d
			break
	if radial_dim is None:
		# Best-effort fallback: assume last dim is radial.
		radial_dim = dims[-1]
	return arr[radial_dim].values


def _center_trace(arr) -> np.ndarray:
	"""Extracts a time trace at the magnetic axis (rho ~ 0)."""
	if "time" not in arr.dims:
		raise ValueError(f"Expected a time-dependent profile; got dims={arr.dims}")
	# Pick the non-time dimension (assume a single radial dimension).
	radial_dims = [d for d in arr.dims if d != "time"]
	if not radial_dims:
		return arr.values
	return arr.isel({radial_dims[0]: 0}).values


def _plot_results(
	*,
	baseline_dt,
	variant_dt,
	baseline_name: str,
	variant_name: str,
	out_dir: pathlib.Path,
	show: bool,
):
	# Headless-safe plotting.
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	t0 = baseline_dt.time.values

	# 1) Center temperature time traces
	Ti0 = _center_trace(baseline_dt.profiles.T_i)
	Te0 = _center_trace(baseline_dt.profiles.T_e)
	Ti1 = _center_trace(variant_dt.profiles.T_i)
	Te1 = _center_trace(variant_dt.profiles.T_e)

	fig1 = plt.figure(figsize=(8, 4))
	plt.plot(t0, Ti0, label=f"{baseline_name}: $T_i$(rho≈0)")
	plt.plot(t0, Te0, label=f"{baseline_name}: $T_e$(rho≈0)")
	plt.plot(variant_dt.time.values, Ti1, "--", label=f"{variant_name}: $T_i$(rho≈0)")
	plt.plot(variant_dt.time.values, Te1, "--", label=f"{variant_name}: $T_e$(rho≈0)")
	plt.xlabel("Time [s]")
	plt.ylabel("Temperature [keV]")
	plt.title("Center temperatures vs time")
	plt.grid(True, alpha=0.25)
	plt.legend()
	fig1.tight_layout()
	fig1_path = out_dir / "fig_center_temperatures.png"
	fig1.savefig(fig1_path, dpi=160)

	# 2) q-profile at final time (comparison)
	q0 = baseline_dt.profiles.q.isel(time=-1)
	q1 = variant_dt.profiles.q.isel(time=-1)
	rho_q0 = _radial_coord(q0)
	rho_q1 = _radial_coord(q1)

	fig2 = plt.figure(figsize=(8, 4))
	plt.plot(rho_q0, q0.values, label=f"{baseline_name}: q(rho), final")
	plt.plot(rho_q1, q1.values, "--", label=f"{variant_name}: q(rho), final")
	plt.xlabel("Normalized radius (rho)")
	plt.ylabel("Safety factor q")
	plt.title("Final safety factor profile")
	plt.grid(True, alpha=0.25)
	plt.legend()
	fig2.tight_layout()
	fig2_path = out_dir / "fig_q_profile_final.png"
	fig2.savefig(fig2_path, dpi=160)

	# 3) Baseline Ti(rho) at a few times
	ti = baseline_dt.profiles.T_i
	rho_ti = _radial_coord(ti)
	t_vals = baseline_dt.time.values
	time_indices = [0, int(len(t_vals) * 0.5), len(t_vals) - 1]
	fig3 = plt.figure(figsize=(8, 4))
	for idx in time_indices:
		plt.plot(rho_ti, ti.isel(time=idx).values, label=f"t={t_vals[idx]:.3f}s")
	plt.xlabel("Normalized radius (rho)")
	plt.ylabel(r"$T_i$ [keV]")
	plt.title(f"{baseline_name}: ion temperature profiles")
	plt.grid(True, alpha=0.25)
	plt.legend()
	fig3.tight_layout()
	fig3_path = out_dir / "fig_ti_profiles.png"
	fig3.savefig(fig3_path, dpi=160)

	# 4) Density profiles in space: n_e(rho) and (if available) n_i(rho)
	fig4_path = None
	profiles0 = baseline_dt.profiles
	profiles1 = variant_dt.profiles
	if ("n_e" in profiles0) and ("n_e" in profiles1):
		ne0 = profiles0.n_e
		ne1 = profiles1.n_e
		rho_ne = _radial_coord(ne0)
		fig4 = plt.figure(figsize=(8, 4))
		for idx in time_indices:
			plt.plot(
				rho_ne,
				ne0.isel(time=idx).values / 1.0e20,
				label=f"{baseline_name}: t={t_vals[idx]:.3f}s",
			)
		# Overlay the final-time variant for a simple comparison.
		plt.plot(
			_radial_coord(ne1.isel(time=-1)),
			ne1.isel(time=-1).values / 1.0e20,
			"--",
			label=f"{variant_name}: final",
		)

		# Optional: ion density, if present.
		if ("n_i" in profiles0) and ("n_i" in profiles1):
			ni0 = profiles0.n_i
			ni1 = profiles1.n_i
			plt.plot(
				_radial_coord(ni0.isel(time=-1)),
				ni0.isel(time=-1).values / 1.0e20,
				":",
				label=f"{baseline_name}: $n_i$ final",
			)
			plt.plot(
				_radial_coord(ni1.isel(time=-1)),
				ni1.isel(time=-1).values / 1.0e20,
				"-.",
				label=f"{variant_name}: $n_i$ final",
			)

		plt.xlabel("Normalized radius (rho)")
		plt.ylabel(r"Density $[10^{20}\,m^{-3}]$")
		plt.title("Density profiles vs radius")
		plt.grid(True, alpha=0.25)
		plt.legend(fontsize=8)
		fig4.tight_layout()
		fig4_path = out_dir / "fig_density_profiles.png"
		fig4.savefig(fig4_path, dpi=160)

	if show:
		# If you want interactive windows, you can re-run with --show and
		# remove the Agg backend line above.
		# In many Linux/headless setups, --show will do nothing.
		plt.show()

	figs = [fig1_path, fig2_path, fig3_path]
	if fig4_path is not None:
		figs.append(fig4_path)
	return figs


def main() -> None:
	_require_script_execution()

	parser = argparse.ArgumentParser(description="TORAX minimal programmatic demo")
	parser.add_argument(
		"--out-dir",
		type=pathlib.Path,
		default=pathlib.Path("torax_outputs"),
		help="Where to write .nc outputs and .png figures",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Attempt to display plots interactively (often not available on headless setups)",
	)
	parser.add_argument(
		"--log-timestep-info",
		action="store_true",
		help="Print per-timestep information from the solver",
	)
	parser.add_argument(
		"--progress-bar",
		action="store_true",
		help="Show a progress bar during the run",
	)
	args = parser.parse_args()

	out_dir: pathlib.Path = args.out_dir
	out_dir.mkdir(parents=True, exist_ok=True)

	base = _build_base_config_dict()

	baseline_name = "baseline_ip_3MA"
	variant_name = "variant_ip_5MA"

	baseline_cfg = dict(base)
	baseline_cfg["profile_conditions"] = dict(base.get("profile_conditions", {}))
	baseline_cfg["profile_conditions"]["Ip"] = 3.0e6

	variant_cfg = dict(base)
	variant_cfg["profile_conditions"] = dict(base.get("profile_conditions", {}))
	variant_cfg["profile_conditions"]["Ip"] = 5.0e6

	baseline_dt, _, baseline_file = _run_case(
		name=baseline_name,
		config_dict=baseline_cfg,
		out_dir=out_dir,
		log_timestep_info=args.log_timestep_info,
		progress_bar=args.progress_bar,
	)
	variant_dt, _, variant_file = _run_case(
		name=variant_name,
		config_dict=variant_cfg,
		out_dir=out_dir,
		log_timestep_info=args.log_timestep_info,
		progress_bar=args.progress_bar,
	)

	fig_paths = _plot_results(
		baseline_dt=baseline_dt,
		variant_dt=variant_dt,
		baseline_name=baseline_name,
		variant_name=variant_name,
		out_dir=out_dir,
		show=args.show,
	)

	print("Wrote TORAX outputs:")
	print(f"  {baseline_file}")
	print(f"  {variant_file}")
	print("Wrote figures:")
	for p in fig_paths:
		print(f"  {p}")


if __name__ == "__main__":
	main()

