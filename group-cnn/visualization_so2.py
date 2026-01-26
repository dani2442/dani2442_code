"""
SO(2) Equivariance Visualization
=================================

Creates a GIF showing that group convolution on SO(2) is equivariant:
- Left: A rotating signal on the circle
- Right: The convolution result with rotation correction (should be static)

This demonstrates that rotating the input, then convolving, gives the same
result as convolving first, then rotating the output.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# =========================================================
# Setup SO(2) discretization
# =========================================================

N = 64
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
dtheta = 2 * np.pi / N

# Create a signal on the circle (asymmetric for visual clarity)
f_original = np.sin(theta) + 0.5 * np.sin(2 * theta) + 0.3 * np.cos(3 * theta)
f_original = f_original - f_original.min()  # Make positive
f_original = f_original / f_original.max()  # Normalize to [0, 1]

# Create a localized kernel (Gaussian-like bump)
L = np.exp(-0.5 * (((theta + np.pi) % (2 * np.pi) - np.pi) / 0.3) ** 2)
L = L / (np.sum(L) * dtheta / (2 * np.pi))  # Normalize


def convolve_so2(f, L):
    """Compute group convolution on SO(2) using FFT."""
    f_hat = np.fft.fft(f)
    L_hat = np.fft.fft(L)
    conv = np.fft.ifft(f_hat * L_hat).real * dtheta / (2 * np.pi)
    return conv


def rotate_signal(f, shift_steps):
    """Rotate signal by shifting indices (left translation on SO(2))."""
    return np.roll(f, shift_steps)


# Compute convolution of original signal
conv_original = convolve_so2(f_original, L)
conv_original = conv_original - conv_original.min()
conv_original = conv_original / conv_original.max()


# =========================================================
# Visualization functions
# =========================================================

def signal_to_radius(signal, base_radius=1.0, amplitude=0.3):
    """Convert signal values to radii for polar plot."""
    return base_radius + amplitude * (signal - 0.5)


def create_figure():
    """Create the figure with two polar subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), 
                                    subplot_kw={'projection': 'polar'})
    
    fig.suptitle('SO(2) Group Convolution Equivariance', fontsize=14, fontweight='bold')
    
    ax1.set_title('Input: Rotating Signal f(θ - α)', fontsize=11)
    ax2.set_title('Output: Conv(Rotated f) with Rotation Correction\n(Should be static)', fontsize=11)
    
    for ax in [ax1, ax2]:
        ax.set_ylim(0, 1.5)
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)
    
    return fig, ax1, ax2


def plot_signal_on_circle(ax, signal, theta, color='blue', label=None):
    """Plot a signal as a filled polar curve."""
    radii = signal_to_radius(signal)
    
    # Close the curve
    theta_closed = np.append(theta, theta[0])
    radii_closed = np.append(radii, radii[0])
    
    # Plot
    line, = ax.plot(theta_closed, radii_closed, color=color, linewidth=2, label=label)
    fill = ax.fill(theta_closed, radii_closed, color=color, alpha=0.3)
    
    return line, fill


# =========================================================
# Animation
# =========================================================

def create_animation():
    """Create the equivariance demonstration animation."""
    fig, ax1, ax2 = create_figure()
    
    n_frames = 60
    rotation_steps = np.linspace(0, N, n_frames, endpoint=False, dtype=int)
    
    # Initial plot elements
    line1, fill1 = plot_signal_on_circle(ax1, f_original, theta, color='#2E86AB')
    line2, fill2 = plot_signal_on_circle(ax2, conv_original, theta, color='#A23B72')
    
    # Reference circle
    ref_circle1 = plt.Circle((0, 0), 1.0, transform=ax1.transData + ax1.transProjectionAffine + ax1.transAxes,
                              fill=False, color='gray', linestyle='--', alpha=0.5)
    ref_circle2 = plt.Circle((0, 0), 1.0, transform=ax2.transData + ax2.transProjectionAffine + ax2.transAxes,
                              fill=False, color='gray', linestyle='--', alpha=0.5)
    
    # Add rotation angle text
    angle_text = fig.text(0.5, 0.02, 'Rotation: α = 0°', ha='center', fontsize=11)
    
    def update(frame):
        shift = rotation_steps[frame]
        rotation_angle = (shift / N) * 360
        
        # Rotate input signal
        f_rotated = rotate_signal(f_original, shift)
        
        # Convolve rotated signal
        conv_rotated = convolve_so2(f_rotated, L)
        conv_rotated = conv_rotated - conv_rotated.min()
        conv_rotated = conv_rotated / (conv_rotated.max() + 1e-10)
        
        # Apply inverse rotation to convolution result (equivariance correction)
        # If convolution is equivariant: Conv(Rotate(f)) = Rotate(Conv(f))
        # So: Rotate^{-1}(Conv(Rotate(f))) = Conv(f) (should be static!)
        conv_corrected = rotate_signal(conv_rotated, -shift)
        
        # Update left plot (rotating input)
        radii_rotated = signal_to_radius(f_rotated)
        theta_closed = np.append(theta, theta[0])
        radii_closed = np.append(radii_rotated, radii_rotated[0])
        
        line1.set_data(theta_closed, radii_closed)
        # Update fill - remove old fills first
        for coll in ax1.collections[:]:
            coll.remove()
        ax1.fill(theta_closed, radii_closed, color='#2E86AB', alpha=0.3)
        
        # Update right plot (corrected convolution - should stay static)
        radii_corrected = signal_to_radius(conv_corrected)
        radii_corr_closed = np.append(radii_corrected, radii_corrected[0])
        
        line2.set_data(theta_closed, radii_corr_closed)
        for coll in ax2.collections[:]:
            coll.remove()
        ax2.fill(theta_closed, radii_corr_closed, color='#A23B72', alpha=0.3)
        
        # Update angle text
        angle_text.set_text(f'Rotation: α = {rotation_angle:.0f}°')
        
        return line1, line2, angle_text
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
    
    return fig, anim


# =========================================================
# Main
# =========================================================

if __name__ == '__main__':
    print("Creating SO(2) equivariance visualization...")
    
    fig, anim = create_animation()
    
    # Save as GIF
    output_path = '/home/dani/projects/dani2442_code/group-cnn/so2_equivariance.gif'
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer, dpi=100)
    
    print(f"Animation saved to: {output_path}")
    plt.close()
