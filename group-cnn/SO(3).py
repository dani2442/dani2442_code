"""
SO(3) Group Convolution Example
================================

This module demonstrates group convolution on SO(3) using:
- Direct discretized integral (O(N^3) for grid points)
- Fourier / Peter-Weyl approach using Wigner D-matrices

The convolution formula on SO(3):
    (Ff)(g) = ∫_{SO(3)} f(gh^{-1}) L(h) dμ(h)

Using Euler angles (α, β, γ):
    dμ = (1/8π²) sin(β) dα dβ dγ

The Fourier transform uses Wigner D-matrices D^ℓ_{mn}(g), and the
convolution theorem gives:
    F̂f(πₗ) = f̂(πₗ) · L̂(πₗ)

where f̂(πₗ) is a (2ℓ+1) × (2ℓ+1) matrix.
"""

import numpy as np
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation

# =========================================================
# 1) Discretization of SO(3) via Euler angles
# =========================================================

N_alpha = 16   # Grid points for α ∈ [0, 2π)
N_beta = 8     # Grid points for β ∈ [0, π]
N_gamma = 16   # Grid points for γ ∈ [0, 2π)

alpha = np.linspace(0, 2*np.pi, N_alpha, endpoint=False)
beta = np.linspace(0, np.pi, N_beta, endpoint=False) + np.pi/(2*N_beta)  # Avoid poles
gamma = np.linspace(0, 2*np.pi, N_gamma, endpoint=False)

d_alpha = 2 * np.pi / N_alpha
d_beta = np.pi / N_beta
d_gamma = 2 * np.pi / N_gamma

# Create meshgrid for Euler angles
Alpha, Beta, Gamma = np.meshgrid(alpha, beta, gamma, indexing='ij')

# Haar measure weight: (1/8π²) sin(β)
haar_weight = np.sin(Beta) / (8 * np.pi**2)
dV = d_alpha * d_beta * d_gamma  # Volume element


# =========================================================
# 2) Helper functions for SO(3) operations
# =========================================================

def euler_to_rotation(a, b, g):
    """Convert Euler angles (ZYZ convention) to rotation matrix."""
    return Rotation.from_euler('ZYZ', [a, b, g]).as_matrix()


def rotation_to_euler(R):
    """Convert rotation matrix to Euler angles (ZYZ convention)."""
    return Rotation.from_matrix(R).as_euler('ZYZ')


def find_nearest_grid_index(a, b, g):
    """Find the nearest grid indices for given Euler angles."""
    # Wrap angles to proper ranges
    a = a % (2 * np.pi)
    g = g % (2 * np.pi)
    b = np.clip(b, 0, np.pi)
    
    i_a = int(np.round(a / d_alpha)) % N_alpha
    i_b = int(np.clip(np.round((b - np.pi/(2*N_beta)) / d_beta), 0, N_beta - 1))
    i_g = int(np.round(g / d_gamma)) % N_gamma
    
    return i_a, i_b, i_g


def wigner_d_small(l, beta):
    """
    Compute the small Wigner d-matrix d^l_{m,n}(β) for all m,n in [-l, l].
    Uses the formula involving Jacobi polynomials / direct recursion.
    Returns a (2l+1, 2l+1) matrix.
    """
    d = np.zeros((2*l + 1, 2*l + 1), dtype=np.float64)
    
    c = np.cos(beta / 2)
    s = np.sin(beta / 2)
    
    for m in range(-l, l + 1):
        for n in range(-l, l + 1):
            # Using the general formula for Wigner d-matrix
            s_min = max(0, n - m)
            s_max = min(l + n, l - m)
            
            val = 0.0
            for ss in range(s_min, s_max + 1):
                num = np.sqrt(
                    np.emath.factorial(l + n) * np.emath.factorial(l - n) *
                    np.emath.factorial(l + m) * np.emath.factorial(l - m)
                )
                den = (
                    np.emath.factorial(l + n - ss) * np.emath.factorial(ss) *
                    np.emath.factorial(l - m - ss) * np.emath.factorial(ss + m - n)
                )
                
                power_c = 2*l + n - m - 2*ss
                power_s = 2*ss + m - n
                
                sign = (-1)**(ss + m - n)
                val += sign * (num / den) * (c**power_c) * (s**power_s)
            
            d[m + l, n + l] = val
    
    return d


def wigner_D_matrix(l, alpha, beta, gamma):
    """
    Compute the Wigner D-matrix D^l_{m,n}(α, β, γ) for all m,n in [-l, l].
    
    D^l_{m,n}(α, β, γ) = e^{-i m α} d^l_{m,n}(β) e^{-i n γ}
    
    Returns a (2l+1, 2l+1) complex matrix.
    """
    d = wigner_d_small(l, beta)
    
    m_vals = np.arange(-l, l + 1)
    n_vals = np.arange(-l, l + 1)
    
    phase_alpha = np.exp(-1j * m_vals * alpha)
    phase_gamma = np.exp(-1j * n_vals * gamma)
    
    D = phase_alpha[:, np.newaxis] * d * phase_gamma[np.newaxis, :]
    
    return D


# =========================================================
# 3) Define signal and kernel on SO(3)
# =========================================================

def create_signal(Alpha, Beta, Gamma):
    """
    Create a band-limited signal on SO(3).
    We use a combination of low-order spherical harmonic-like functions.
    """
    # Simple signal based on Euler angles
    f = (np.sin(2 * Beta) * np.cos(Alpha) + 
         0.5 * np.cos(Beta) * np.sin(2 * Gamma) +
         0.3 * np.sin(Alpha + Gamma))
    return f


def create_kernel(Alpha, Beta, Gamma, sigma=0.5):
    """
    Create a localized kernel on SO(3).
    A Gaussian-like bump centered at the identity (α=0, β=0, γ=0).
    """
    # Distance from identity in terms of rotation angle
    # For small angles, β is the main contributor
    dist_sq = Beta**2 + 0.5 * (Alpha**2 + Gamma**2) * (1 - np.cos(Beta))
    L = np.exp(-dist_sq / (2 * sigma**2))
    return L


# Create signal and kernel
f = create_signal(Alpha, Beta, Gamma)
L = create_kernel(Alpha, Beta, Gamma, sigma=0.5)

# Normalize kernel
L = L / (np.sum(L * haar_weight) * dV + 1e-12)


# =========================================================
# 4) Method A: Direct discretized integral (O(N^6) naive)
# =========================================================
print("Computing direct convolution (this may take a moment)...")

conv_direct = np.zeros_like(f)

# For efficiency, we precompute all rotation matrices
rotations = np.zeros((N_alpha, N_beta, N_gamma, 3, 3))
for i_a in range(N_alpha):
    for i_b in range(N_beta):
        for i_g in range(N_gamma):
            rotations[i_a, i_b, i_g] = euler_to_rotation(
                alpha[i_a], beta[i_b], gamma[i_g]
            )

# Direct convolution: (Ff)(g) = ∫ f(gh^{-1}) L(h) dμ(h)
for i_a in range(N_alpha):
    for i_b in range(N_beta):
        for i_g in range(N_gamma):
            R_g = rotations[i_a, i_b, i_g]
            
            integral = 0.0
            for j_a in range(N_alpha):
                for j_b in range(N_beta):
                    for j_g in range(N_gamma):
                        R_h = rotations[j_a, j_b, j_g]
                        
                        # Compute g * h^{-1}
                        R_gh_inv = R_g @ R_h.T
                        
                        # Find Euler angles of result
                        euler_gh_inv = rotation_to_euler(R_gh_inv)
                        
                        # Find nearest grid point
                        k_a, k_b, k_g = find_nearest_grid_index(*euler_gh_inv)
                        
                        # Accumulate integral
                        integral += (f[k_a, k_b, k_g] * L[j_a, j_b, j_g] * 
                                   haar_weight[j_a, j_b, j_g])
            
            conv_direct[i_a, i_b, i_g] = integral * dV

print("Direct convolution complete.")


# =========================================================
# 5) Method B: Fourier / Peter-Weyl using Wigner D-matrices
# =========================================================
print("Computing FFT-based convolution...")

L_MAX = 4  # Maximum ℓ for the expansion (determines accuracy)

def compute_fourier_coefficient(func, l):
    """
    Compute the Fourier coefficient f̂(πₗ), a (2l+1) × (2l+1) matrix.
    
    f̂(πₗ)_{mn} = ∫_{SO(3)} f(g) D^l_{mn}(g^{-1}) dμ(g)
                = ∫_{SO(3)} f(g) conj(D^l_{nm}(g)) dμ(g)
    """
    f_hat = np.zeros((2*l + 1, 2*l + 1), dtype=np.complex128)
    
    for i_a in range(N_alpha):
        for i_b in range(N_beta):
            for i_g in range(N_gamma):
                D = wigner_D_matrix(l, alpha[i_a], beta[i_b], gamma[i_g])
                weight = haar_weight[i_a, i_b, i_g] * dV
                
                # f̂_{mn} = ∫ f(g) conj(D^l_{nm}(g)) dμ
                f_hat += func[i_a, i_b, i_g] * np.conj(D.T) * weight
    
    return f_hat


def inverse_fourier(f_hats, L_max):
    """
    Reconstruct function from Fourier coefficients using Peter-Weyl:
    f(g) = Σ_ℓ (2ℓ+1) Tr[f̂(πₗ) D^ℓ(g)]
    """
    result = np.zeros((N_alpha, N_beta, N_gamma), dtype=np.complex128)
    
    for i_a in range(N_alpha):
        for i_b in range(N_beta):
            for i_g in range(N_gamma):
                val = 0.0
                for l in range(L_max + 1):
                    D = wigner_D_matrix(l, alpha[i_a], beta[i_b], gamma[i_g])
                    val += (2*l + 1) * np.trace(f_hats[l] @ D)
                result[i_a, i_b, i_g] = val
    
    return result.real


# Compute Fourier coefficients
f_hats = {}
L_hats = {}
for l in range(L_MAX + 1):
    f_hats[l] = compute_fourier_coefficient(f, l)
    L_hats[l] = compute_fourier_coefficient(L, l)

# Convolution in Fourier domain: (Ff)^ = f̂ · L̂
conv_hats = {}
for l in range(L_MAX + 1):
    conv_hats[l] = f_hats[l] @ L_hats[l]

# Inverse transform
conv_fft = inverse_fourier(conv_hats, L_MAX)

print("FFT convolution complete.")


# =========================================================
# 6) Check agreement between methods
# =========================================================
max_err = np.max(np.abs(conv_direct - conv_fft))
print(f"\nMax |direct - fft| = {max_err:.6e}")
print("(Note: Error depends on grid resolution and L_MAX)")


# =========================================================
# 7) Equivariance check
# =========================================================
print("\n--- Equivariance Check ---")

# Define a rotation to apply (a shift in SO(3))
shift_euler = (np.pi/3, np.pi/4, np.pi/6)  # (α_s, β_s, γ_s)
R_shift = euler_to_rotation(*shift_euler)

# Create shifted input: (λ(R_s)f)(g) = f(R_s^{-1} g)
f_shifted = np.zeros_like(f)
for i_a in range(N_alpha):
    for i_b in range(N_beta):
        for i_g in range(N_gamma):
            R_g = rotations[i_a, i_b, i_g]
            R_new = R_shift.T @ R_g  # R_s^{-1} * g
            euler_new = rotation_to_euler(R_new)
            k_a, k_b, k_g = find_nearest_grid_index(*euler_new)
            f_shifted[i_a, i_b, i_g] = f[k_a, k_b, k_g]

# Convolve shifted input
print("Computing convolution of shifted input...")
conv_shifted_input = np.zeros_like(f)

for i_a in range(N_alpha):
    for i_b in range(N_beta):
        for i_g in range(N_gamma):
            R_g = rotations[i_a, i_b, i_g]
            
            integral = 0.0
            for j_a in range(N_alpha):
                for j_b in range(N_beta):
                    for j_g in range(N_gamma):
                        R_h = rotations[j_a, j_b, j_g]
                        R_gh_inv = R_g @ R_h.T
                        euler_gh_inv = rotation_to_euler(R_gh_inv)
                        k_a, k_b, k_g = find_nearest_grid_index(*euler_gh_inv)
                        
                        integral += (f_shifted[k_a, k_b, k_g] * L[j_a, j_b, j_g] * 
                                   haar_weight[j_a, j_b, j_g])
            
            conv_shifted_input[i_a, i_b, i_g] = integral * dV

# Shift the convolved output: (λ(R_s)(Ff))(g)
conv_output_shifted = np.zeros_like(f)
for i_a in range(N_alpha):
    for i_b in range(N_beta):
        for i_g in range(N_gamma):
            R_g = rotations[i_a, i_b, i_g]
            R_new = R_shift.T @ R_g
            euler_new = rotation_to_euler(R_new)
            k_a, k_b, k_g = find_nearest_grid_index(*euler_new)
            conv_output_shifted[i_a, i_b, i_g] = conv_direct[k_a, k_b, k_g]

# Check equivariance: F(λ(g)f) should equal λ(g)(Ff)
equiv_err = np.max(np.abs(conv_shifted_input - conv_output_shifted))
print(f"Equivariance error = {equiv_err:.6e}")
print("(Error is due to discretization; decreases with finer grids)")
