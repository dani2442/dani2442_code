"""
SO(2) Equivariance Visualization
=================================

Creates a GIF showing that group convolution on SO(2) is equivariant:
  rotate(f) * kernel = rotate(output)

Three panels:
  Left   – rotating input signal
  Centre – static convolution kernel
  Right  – un-rotated convolution result  (should stay constant)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =========================================================
# Setup SO(2) discretization
# =========================================================

N = 64
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
dtheta = 2 * np.pi / N


# =========================================================
# 1-D periodic Perlin-like noise
# =========================================================

def perlin_noise_1d(n, octaves=5, seed=42):
    """Generate 1-D periodic Perlin-like noise (fBm) on a circle."""
    rng = np.random.default_rng(seed)
    result = np.zeros(n)
    for o in range(octaves):
        freq = 2 ** (o + 1)
        amp = 0.5 ** o
        gradients = rng.uniform(-1, 1, freq)

        x = np.linspace(0, freq, n, endpoint=False)
        xi = np.floor(x).astype(int) % freq
        xf = x - np.floor(x)

        u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
        result += amp * ((1 - u) * gradients[xi] + u * gradients[(xi + 1) % freq])
    return result


# ------ signals ------
f_original = perlin_noise_1d(N, octaves=5, seed=42)
f_original = (f_original - f_original.min()) / (f_original.max() - f_original.min())

# Localised kernel (Gaussian-like bump)
L = np.exp(-0.5 * (((theta + np.pi) % (2 * np.pi) - np.pi) / 0.3) ** 2)
L = L / (np.sum(L) * dtheta / (2 * np.pi))
# Normalise kernel to [0, 1] for display
L_display = (L - L.min()) / (L.max() - L.min())


def convolve_so2(f, L):
    """Compute group convolution on SO(2) using FFT."""
    f_hat = np.fft.fft(f)
    L_hat = np.fft.fft(L)
    return np.fft.ifft(f_hat * L_hat).real * dtheta / (2 * np.pi)


def rotate_signal(f, shift_steps):
    """Rotate signal by shifting indices (left translation on SO(2))."""
    return np.roll(f, shift_steps)


conv_original = convolve_so2(f_original, L)
conv_original = (conv_original - conv_original.min()) / \
                (conv_original.max() - conv_original.min())


# =========================================================
# Visualization helpers
# =========================================================

def signal_to_radius(signal, base_radius=1.0, amplitude=0.3):
    return base_radius + amplitude * (signal - 0.5)


TERRAIN_CMAP = plt.colormaps.get_cmap('terrain')


def plot_signal_on_circle(ax, signal, theta_arr):
    """Plot a signal as terrain-coloured bars in polar coordinates."""
    radii = signal_to_radius(signal)
    base = 0

    sig_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-10)
    colors = TERRAIN_CMAP(0.25 + 0.6 * sig_norm)

    ax.bar(theta_arr, radii - base, bottom=base, width=dtheta * 1.05,
           color=colors, alpha=0.9, edgecolor='none')

    theta_closed = np.append(theta_arr, theta_arr[0])
    radii_closed = np.append(radii, radii[0])
    ax.plot(theta_closed, radii_closed, color='#2a2a2a', linewidth=0.8)
    ax.plot(theta_closed, np.full_like(theta_closed, 1.0),
            color='gray', linestyle='--', alpha=0.4, linewidth=0.5)


# =========================================================
# Animation
# =========================================================

def create_animation():
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16, 5), subplot_kw={'projection': 'polar'})

    fig.suptitle('SO(2) Group Convolution Equivariance',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(top=0.80, bottom=0.08, wspace=0.35)

    n_frames = 60
    rotation_steps = np.linspace(0, N, n_frames, endpoint=False, dtype=int)
    angle_text = fig.text(0.5, 0.01, 'Rotation: α = 0°',
                          ha='center', fontsize=11)

    titles = [
        r'Input: $R_\alpha f$',
        r'Kernel $\psi$',
        r'Output: $R_\alpha^{-1}(R_\alpha f * \psi)$'
        '\n(Should match $f * \\psi$)',
    ]

    def setup_axes():
        for ax, title in zip([ax1, ax2, ax3], titles):
            ax.set_ylim(0, 1.5)
            ax.set_yticklabels([])
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=10, pad=18)

    # Initial draw
    plot_signal_on_circle(ax1, f_original, theta)
    plot_signal_on_circle(ax2, L_display, theta)
    plot_signal_on_circle(ax3, conv_original, theta)
    setup_axes()

    def update(frame):
        shift = rotation_steps[frame]
        rotation_angle = (shift / N) * 360

        f_rotated = rotate_signal(f_original, shift)

        conv_rotated = convolve_so2(f_rotated, L)
        conv_rotated = (conv_rotated - conv_rotated.min()) / \
                       (conv_rotated.max() - conv_rotated.min() + 1e-10)
        conv_corrected = rotate_signal(conv_rotated, -shift)

        ax1.clear(); ax2.clear(); ax3.clear()

        plot_signal_on_circle(ax1, f_rotated, theta)
        plot_signal_on_circle(ax2, L_display, theta)
        plot_signal_on_circle(ax3, conv_corrected, theta)
        setup_axes()

        angle_text.set_text(f'Rotation: α = {rotation_angle:.0f}°')
        return []

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=100, blit=False)
    return fig, anim


# =========================================================
# Main
# =========================================================

if __name__ == '__main__':
    print("Creating SO(2) equivariance visualization …")

    fig, anim = create_animation()

    output_path = '/home/dani/projects/dani2442_code/group-cnn/so2_equivariance.gif'
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer, dpi=100)

    print(f"Animation saved to: {output_path}")
    plt.close()
