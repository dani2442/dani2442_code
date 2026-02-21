"""
SO(3) Equivariance Visualization (on S²)
========================================

Creates a GIF showing that SO(3)-equivariant filtering on the sphere commutes
with rotations:

  (R f) ⋆ ψ = R (f ⋆ ψ)

Three panels:
  Left   – rotating input signal on the sphere
  Centre – static (zonal) kernel on the sphere
  Right  – convolution result (rotates with input, demonstrating equivariance)

Implementation notes:
  - The convolution is a *zonal* (isotropic) spherical filter implemented in
    spherical-harmonic space (degree-wise scaling).
  - Rotations are applied in spherical-harmonic space via Wigner D-matrices
    loaded from `group-cnn/SO(3).py`, avoiding the interpolation artefacts that
    made the right panel drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation
from scipy.special import eval_legendre, sph_harm

# =========================================================
# Configuration
# =========================================================

N_PHI = 60
N_THETA = 30

L_MAX = 12
KERNEL_SIGMA = 0.45

N_FRAMES = 60
ROTATION_AXIS = np.array([0.5, 0.5, 1.0], dtype=np.float64)

# Visuals
TERRAIN_CMAP = plt.colormaps.get_cmap("terrain")
DISPLACEMENT = 0.08  # radial bump for landform height
FIXED_VIEW = dict(elev=25, azim=45)
LIM = 1.15


# =========================================================
# Setup sphere discretization (equiangular grid)
# =========================================================

phi = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
theta_sphere = np.linspace(0.0, np.pi, N_THETA)

Phi, Theta = np.meshgrid(phi, theta_sphere)

X = np.sin(Theta) * np.cos(Phi)
Y = np.sin(Theta) * np.sin(Phi)
Z = np.cos(Theta)

# For plotting, close the seam in φ by duplicating the first column.
X_PLOT = np.concatenate([X, X[:, :1]], axis=1)
Y_PLOT = np.concatenate([Y, Y[:, :1]], axis=1)
Z_PLOT = np.concatenate([Z, Z[:, :1]], axis=1)

DPHI = 2 * np.pi / N_PHI
DTHETA = np.pi / (N_THETA - 1)
WEIGHTS = np.sin(Theta) * DTHETA * DPHI
WEIGHTS_FLAT = WEIGHTS.reshape(-1)

PHI_FLAT = Phi.reshape(-1)
THETA_FLAT = Theta.reshape(-1)


# =========================================================
# Load Wigner D from SO(3).py (file name isn't importable)
# =========================================================

_SO3_PATH = Path(__file__).with_name("SO(3).py")
_SO3_SPEC = importlib.util.spec_from_file_location("_so3_group", _SO3_PATH)
if _SO3_SPEC is None or _SO3_SPEC.loader is None:
    raise RuntimeError(f"Failed to load module spec for: {_SO3_PATH}")
_SO3 = importlib.util.module_from_spec(_SO3_SPEC)
_SO3_SPEC.loader.exec_module(_SO3)

wigner_D_matrix = _SO3.wigner_D_matrix


# =========================================================
# Spherical harmonics basis + transforms (truncated to ℓ ≤ L_MAX)
# =========================================================

def precompute_sph_harm_basis(
    phi_flat: np.ndarray, theta_flat: np.ndarray, l_max: int
) -> dict[int, np.ndarray]:
    """
    Basis Y^ℓ_m evaluated on the sampling grid.

    Returns dict: ℓ -> array of shape (P, 2ℓ+1) with m ordered from -ℓ..ℓ.
    """
    basis: dict[int, np.ndarray] = {}
    for l in range(l_max + 1):
        Y_l = np.empty((phi_flat.size, 2 * l + 1), dtype=np.complex128)
        for col, m in enumerate(range(-l, l + 1)):
            Y_l[:, col] = sph_harm(m, l, phi_flat, theta_flat)
        basis[l] = Y_l
    return basis


Y_BASIS = precompute_sph_harm_basis(PHI_FLAT, THETA_FLAT, L_MAX)


def sht_forward(samples: np.ndarray) -> dict[int, np.ndarray]:
    """Compute truncated spherical-harmonic coefficients for a real signal."""
    f_w = samples.reshape(-1) * WEIGHTS_FLAT
    coeffs: dict[int, np.ndarray] = {}
    for l, Y_l in Y_BASIS.items():
        coeffs[l] = Y_l.conj().T @ f_w
    return coeffs


def sht_inverse(coeffs: dict[int, np.ndarray]) -> np.ndarray:
    """Invert truncated spherical-harmonic coefficients back to the grid."""
    out = np.zeros(PHI_FLAT.shape, dtype=np.complex128)
    for l, Y_l in Y_BASIS.items():
        out += Y_l @ coeffs[l]
    return out.reshape(Theta.shape).real


def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    return (arr - lo) / (hi - lo + 1e-12)


# =========================================================
# 3-D value noise for rough terrain on the sphere
# =========================================================

def value_noise_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, octaves: int = 5, seed: int = 42) -> np.ndarray:
    """fBm value noise evaluated at sphere-surface coordinates."""
    shape = X.shape
    result = np.zeros(shape)
    coords = np.stack([X + 2.0, Y + 2.0, Z + 2.0], axis=-1)

    for o in range(octaves):
        freq = 2 ** (o + 1)
        amp = 0.55 ** o

        sc = coords * freq
        gi = np.floor(sc).astype(np.int64)
        gf = sc - np.floor(sc)
        u = gf * gf * gf * (gf * (gf * 6 - 15) + 10)

        val = np.zeros(shape)
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    ix = gi[..., 0] + dx
                    iy = gi[..., 1] + dy
                    iz = gi[..., 2] + dz

                    h = ix * 374761393 + iy * 668265263 + iz * 1274126177
                    h = h + np.int64(seed + o * 7919)
                    h = np.abs(h)
                    corner_val = (h % 999983).astype(float) / 999983.0

                    wx = u[..., 0] if dx else (1 - u[..., 0])
                    wy = u[..., 1] if dy else (1 - u[..., 1])
                    wz = u[..., 2] if dz else (1 - u[..., 2])

                    val += corner_val * wx * wy * wz
        result += amp * (val - 0.5)
    return result


def create_bandlimited_signal() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """
    Create a terrain-like signal, then project to the truncated ℓ≤L_MAX space.

    Returns (f_original in [0,1], f_coeffs).
    """
    f = normalize_01(value_noise_3d(X, Y, Z, octaves=5, seed=42))

    # Project to the truncated harmonic space and re-normalize for plotting.
    coeffs = sht_forward(f)
    f_band = normalize_01(sht_inverse(coeffs))
    return f_band, sht_forward(f_band)


# =========================================================
# Zonal kernel + convolution eigenvalues (Funk–Hecke)
# =========================================================

def create_zonal_kernel(theta: np.ndarray, sigma: float) -> np.ndarray:
    """
    Zonal kernel k(θ) centered at the north pole, normalized so:
      ∫_{S²} k dΩ ≈ 1
    """
    k = np.exp(-(theta**2) / (2 * sigma**2))
    norm = 2 * np.pi * np.sum(k * np.sin(theta)) * DTHETA
    return k / (norm + 1e-12)


def zonal_eigenvalues(k_theta: np.ndarray, theta: np.ndarray, l_max: int) -> np.ndarray:
    """
    Eigenvalues λ_ℓ for the integral operator:
      (Tf)(x) = ∫ k(angle(x,y)) f(y) dΩ_y

    λ_ℓ = 2π ∫_0^π k(θ) P_ℓ(cosθ) sinθ dθ
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    lambdas = np.empty(l_max + 1, dtype=np.float64)
    for l in range(l_max + 1):
        P_l = eval_legendre(l, cos_t)
        lambdas[l] = 2 * np.pi * np.sum(k_theta * P_l * sin_t) * DTHETA
    return lambdas


# =========================================================
# Build signal, kernel, and filtered output
# =========================================================

print("Precomputing signal/kernel …")
f_original, f_coeffs = create_bandlimited_signal()

k_theta = create_zonal_kernel(theta_sphere, sigma=KERNEL_SIGMA)
kernel_display = normalize_01(k_theta[:, None] * np.ones((1, N_PHI), dtype=np.float64))

lambdas = zonal_eigenvalues(k_theta, theta_sphere, L_MAX)

conv_coeffs = {l: lambdas[l] * f_coeffs[l] for l in range(L_MAX + 1)}
conv_original = sht_inverse(conv_coeffs)
conv_display = normalize_01(conv_original)


# =========================================================
# Precompute rotated inputs (in coefficient space)
# =========================================================

print("Precomputing rotations …")
ROTATION_AXIS /= np.linalg.norm(ROTATION_AXIS)
angles = np.linspace(0.0, 2 * np.pi, N_FRAMES, endpoint=False)
rotations = [Rotation.from_rotvec(a * ROTATION_AXIS) for a in angles]

f_rotated_frames: list[np.ndarray] = []
conv_rotated_frames: list[np.ndarray] = []
max_coeff_err = 0.0

for angle, rot in zip(angles, rotations):
    # The identity rotation triggers a ZYZ gimbal-lock warning in SciPy; for
    # that case the Wigner D matrices are just identities.
    if abs(float(angle)) < 1e-12:
        D_l = {l: np.eye(2 * l + 1, dtype=np.complex128) for l in range(L_MAX + 1)}
    else:
        alpha, beta, gamma = rot.as_euler("ZYZ", degrees=False)
        D_l = {l: wigner_D_matrix(l, alpha, beta, gamma) for l in range(L_MAX + 1)}

    # Rotate signal coefficients: f_R(x) = f(R^{-1} x)
    f_rot_coeffs = {l: (D_l[l] @ f_coeffs[l]) for l in range(L_MAX + 1)}
    f_rot = sht_inverse(f_rot_coeffs)
    f_rotated_frames.append(np.clip(f_rot, 0.0, 1.0))

    # Compute convolution of rotated signal (without un-rotation)
    conv_rot_coeffs = {l: lambdas[l] * f_rot_coeffs[l] for l in range(L_MAX + 1)}
    conv_rot = sht_inverse(conv_rot_coeffs)
    conv_rotated_frames.append(normalize_01(np.clip(conv_rot, conv_rot.min(), conv_rot.max())))

    # Sanity check: D† (λ D f̂) ≈ λ f̂
    conv_corr_coeffs = {l: (D_l[l].conj().T @ conv_rot_coeffs[l]) for l in range(L_MAX + 1)}
    for l in range(L_MAX + 1):
        max_coeff_err = max(max_coeff_err, float(np.max(np.abs(conv_corr_coeffs[l] - conv_coeffs[l]))))

print(f"Max coefficient equivariance error (ℓ≤{L_MAX}): {max_coeff_err:.3e}")


# =========================================================
# Visualization helpers
# =========================================================

def plot_sphere_with_terrain(ax: plt.Axes, signal: np.ndarray, title: str) -> None:
    ax.clear()

    sig = np.clip(signal, 0.0, 1.0)
    sig_plot = np.concatenate([sig, sig[:, :1]], axis=1)

    disp = 1.0 + DISPLACEMENT * sig_plot
    Xd, Yd, Zd = disp * X_PLOT, disp * Y_PLOT, disp * Z_PLOT

    colors = TERRAIN_CMAP(0.20 + 0.65 * sig_plot)
    ax.plot_surface(
        Xd,
        Yd,
        Zd,
        facecolors=colors,
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=True,
    )

    ax.set_xlim([-LIM, LIM])
    ax.set_ylim([-LIM, LIM])
    ax.set_zlim([-LIM, LIM])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=8)
    ax.view_init(**FIXED_VIEW)


# =========================================================
# Animation
# =========================================================

def create_animation() -> tuple[plt.Figure, FuncAnimation]:
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    fig.suptitle("SO(3) Equivariance on the Sphere (zonal filter)", fontsize=14, fontweight="bold", y=0.95)
    fig.subplots_adjust(top=0.88, bottom=0.08, wspace=0.05)

    # Add * and = symbols between panels
    fig.text(0.355, 0.48, r'$\ast$', ha='center', va='center',
             fontsize=28, fontweight='bold')
    fig.text(0.665, 0.48, r'$=$', ha='center', va='center',
             fontsize=28, fontweight='bold')

    angle_text = fig.text(0.5, 0.02, "Rotation angle = 0°", ha="center", fontsize=11)

    def update(frame: int):
        angle = angles[frame]

        plot_sphere_with_terrain(ax1, f_rotated_frames[frame], r"Input: $R f$")
        plot_sphere_with_terrain(ax2, kernel_display, r"Kernel $\psi$")
        plot_sphere_with_terrain(ax3, conv_rotated_frames[frame], r"Output: $R f \star \psi$")

        angle_text.set_text(f"Rotation angle = {np.degrees(angle):.0f}°")
        return []

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, blit=False)
    return fig, anim


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    print("Creating SO(3) equivariance visualization …")

    fig, anim = create_animation()

    output_path = Path(__file__).with_name("so3_equivariance.gif")
    writer = PillowWriter(fps=10)
    anim.save(str(output_path), writer=writer, dpi=100)

    print(f"Animation saved to: {output_path}")
    plt.close(fig)
