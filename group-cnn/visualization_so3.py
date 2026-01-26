"""
SO(3) Equivariance Visualization
=================================

Creates a GIF showing that group convolution on SO(3) is equivariant:
- Left: A rotating signal on a sphere
- Right: The convolution result with rotation correction (should be static)

This demonstrates that rotating the input, then convolving, gives the same
result as convolving first, then rotating the output.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# =========================================================
# Setup sphere discretization
# =========================================================

# Spherical coordinates for visualization
N_phi = 50      # Azimuthal angle points
N_theta = 25    # Polar angle points

phi = np.linspace(0, 2 * np.pi, N_phi)       # Azimuthal [0, 2π]
theta_sphere = np.linspace(0, np.pi, N_theta)  # Polar [0, π]

Phi, Theta = np.meshgrid(phi, theta_sphere)

# Unit sphere coordinates
X = np.sin(Theta) * np.cos(Phi)
Y = np.sin(Theta) * np.sin(Phi)
Z = np.cos(Theta)


# =========================================================
# Signal and kernel on S² (functions on the sphere)
# =========================================================

def create_signal_on_sphere(Phi, Theta):
    """
    Create an asymmetric signal on S² using spherical harmonics-like functions.
    This gives a clear visual pattern that shows rotation.
    """
    # Combination of low-order spherical harmonics (real forms)
    Y_10 = np.cos(Theta)  # l=1, m=0
    Y_11 = np.sin(Theta) * np.cos(Phi)  # l=1, m=1 (real)
    Y_20 = (3 * np.cos(Theta)**2 - 1) / 2  # l=2, m=0
    Y_21 = np.sin(Theta) * np.cos(Theta) * np.cos(Phi)  # l=2, m=1
    Y_22 = np.sin(Theta)**2 * np.cos(2 * Phi)  # l=2, m=2
    
    # Create asymmetric pattern
    f = 0.5 * Y_10 + 0.8 * Y_11 + 0.3 * Y_20 + 0.4 * Y_21 + 0.2 * Y_22
    
    # Normalize to [0, 1]
    f = (f - f.min()) / (f.max() - f.min() + 1e-10)
    
    return f


def create_kernel_on_sphere(Theta, sigma=0.5):
    """
    Create a localized kernel on S² (Gaussian-like bump at north pole).
    """
    # Distance from north pole (θ=0)
    L = np.exp(-Theta**2 / (2 * sigma**2))
    
    # Normalize
    L = L / (L.sum() + 1e-10)
    
    return L


def convolve_on_sphere(f, L, Phi, Theta):
    """
    Simplified spherical convolution using FFT in the azimuthal direction.
    This is an approximation for visualization purposes.
    """
    # For each latitude, convolve in the azimuthal direction
    result = np.zeros_like(f)
    
    for i in range(len(theta_sphere)):
        # Weight by kernel at this latitude
        weight = L[i, :].mean()
        
        # FFT-based circular convolution
        f_row = f[i, :]
        L_row = L[i, :]
        
        f_hat = np.fft.fft(f_row)
        L_hat = np.fft.fft(L_row)
        
        conv_row = np.fft.ifft(f_hat * L_hat).real
        result[i, :] = conv_row
    
    # Smooth across latitudes (simple averaging with neighbors)
    result_smooth = result.copy()
    for i in range(1, len(theta_sphere) - 1):
        result_smooth[i, :] = (0.2 * result[i-1, :] + 
                               0.6 * result[i, :] + 
                               0.2 * result[i+1, :])
    
    return result_smooth


def rotate_sphere_signal(f, X, Y, Z, R):
    """
    Rotate a signal on the sphere by applying rotation R to the coordinates
    and interpolating the signal values.
    """
    # Stack coordinates
    coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Apply inverse rotation to coordinates (to get the source position)
    R_inv = R.T
    coords_rotated = coords @ R_inv.T
    
    # Convert back to spherical
    x_rot, y_rot, z_rot = coords_rotated[:, 0], coords_rotated[:, 1], coords_rotated[:, 2]
    
    theta_rot = np.arccos(np.clip(z_rot, -1, 1))
    phi_rot = np.arctan2(y_rot, x_rot) % (2 * np.pi)
    
    # Interpolate signal values (nearest neighbor for simplicity)
    i_theta = np.clip(np.round(theta_rot / np.pi * (N_theta - 1)).astype(int), 0, N_theta - 1)
    i_phi = np.round(phi_rot / (2 * np.pi) * (N_phi - 1)).astype(int) % N_phi
    
    f_rotated = f[i_theta, i_phi].reshape(f.shape)
    
    return f_rotated


# Create original signal and kernel
f_original = create_signal_on_sphere(Phi, Theta)
L = create_kernel_on_sphere(Theta, sigma=0.4)

# Compute convolution of original
conv_original = convolve_on_sphere(f_original, L, Phi, Theta)
conv_original = (conv_original - conv_original.min()) / (conv_original.max() - conv_original.min() + 1e-10)


# =========================================================
# Visualization
# =========================================================

def plot_sphere_with_signal(ax, X, Y, Z, signal, title, cmap='viridis'):
    """Plot a sphere colored by the signal values."""
    ax.clear()
    
    # Normalize signal for colormap
    signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-10)
    
    # Plot surface
    colormap = plt.colormaps.get_cmap(cmap)
    surf = ax.plot_surface(X, Y, Z, facecolors=colormap(signal_norm),
                          rstride=1, cstride=1, shade=False, antialiased=True)
    
    # Set equal aspect ratio
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])
    
    # Remove axes for cleaner look
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=10)
    
    return surf


def create_animation():
    """Create the equivariance demonstration animation."""
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    fig.suptitle('SO(3) Group Convolution Equivariance', fontsize=14, fontweight='bold')
    
    n_frames = 60
    
    # Rotation axis (arbitrary direction for interesting rotation)
    rotation_axis = np.array([0.5, 0.5, 1.0])
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    
    # Add rotation angle text
    angle_text = fig.text(0.5, 0.02, 'Rotation: α = 0°', ha='center', fontsize=11)
    
    def update(frame):
        angle = angles[frame]
        rotation_degrees = np.degrees(angle)
        
        # Create rotation matrix using axis-angle representation
        R = Rotation.from_rotvec(angle * rotation_axis).as_matrix()
        
        # Rotate input signal
        f_rotated = rotate_sphere_signal(f_original, X, Y, Z, R)
        
        # Convolve rotated signal
        conv_rotated = convolve_on_sphere(f_rotated, L, Phi, Theta)
        conv_rotated = (conv_rotated - conv_rotated.min()) / (conv_rotated.max() - conv_rotated.min() + 1e-10)
        
        # Apply inverse rotation to convolution result (equivariance correction)
        # If convolution is equivariant: Conv(Rotate(f)) = Rotate(Conv(f))
        # So: Rotate^{-1}(Conv(Rotate(f))) = Conv(f) (should be static!)
        R_inv = R.T
        conv_corrected = rotate_sphere_signal(conv_rotated, X, Y, Z, R_inv)
        
        # Plot left sphere (rotating input)
        plot_sphere_with_signal(ax1, X, Y, Z, f_rotated, 
                               'Input: Rotating Signal f(R⁻¹ · x)', cmap='plasma')
        
        # Plot right sphere (corrected convolution - should be static)
        plot_sphere_with_signal(ax2, X, Y, Z, conv_corrected,
                               'Output: Conv(Rotated f) with Correction\n(Should be static)', 
                               cmap='viridis')
        
        # Update angle text
        angle_text.set_text(f'Rotation: α = {rotation_degrees:.0f}°')
        
        return []
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
    
    return fig, anim


# =========================================================
# Main
# =========================================================

if __name__ == '__main__':
    print("Creating SO(3) equivariance visualization...")
    print("This may take a few minutes...")
    
    fig, anim = create_animation()
    
    # Save as GIF
    output_path = '/home/dani/projects/dani2442_code/group-cnn/so3_equivariance.gif'
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer, dpi=100)
    
    print(f"Animation saved to: {output_path}")
    plt.close()
