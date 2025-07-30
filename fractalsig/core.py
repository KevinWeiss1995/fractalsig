"""
Core functions for fractional signal analysis.
"""

import numpy as np
import pywt
from scipy.fft import fft as scipy_fft, ifft as scipy_ifft, fftfreq


def fgn(H, L):
    """
    Generate a fractional Gaussian noise (fGn) time series.
    
    Uses Cholesky decomposition for L < 2^20 (more stable, exact).
    Uses Davies-Harte method for L >= 2^20 (more efficient for large datasets).
    
    Parameters:
        H (float): Hurst exponent (0 < H < 1)
        L (int): Length of the time series
        
    Returns:
        np.ndarray: Array of shape (L,) representing fractional Gaussian noise
        
    Raises:
        ValueError: If H is not in (0, 1)
        UserWarning: If L is not a power of two (for Davies-Harte method)
    """
    if not (0 < H < 1):
        raise ValueError(f'Hurst exponent H must be in (0, 1), got {H}')
    
    if L <= 0:
        raise ValueError(f'Length L must be positive, got {L}')
    
    # Handle edge case for very small lengths
    if L == 1:
        return np.array([np.random.randn()])
    
    # Use Cholesky method as primary approach for L < 2^20 (1,048,576)
    # This is more numerically stable and exact
    if L < 2**20:
        return _fgn_simple(H, L)
    
    # For very large datasets (L >= 2^20), use Davies-Harte for efficiency
    # Check if L is a power of two and warn if not
    if L > 1 and (L & (L - 1)) != 0:
        import warnings
        # Find the next power of two for suggestion
        next_power_of_two = 1 << (L - 1).bit_length()
        prev_power_of_two = next_power_of_two >> 1
        
        warnings.warn(
            f'Length L={L} is not a power of two. The Davies-Harte method uses FFT operations '
            f'which are most efficient with power-of-two lengths (e.g., {prev_power_of_two}, {next_power_of_two}). '
            f'Non-power-of-two lengths will result in slower FFT computations and may cause '
            f'the circulant embedding matrix to have a larger size (2*(L-1)={2*(L-1)}), '
            f'potentially leading to increased memory usage and computation time. '
            f'Consider using a nearby power of two for optimal performance.',
            UserWarning,
            stacklevel=2
        )
    
    # Davies-Harte method for large datasets
    # Reference: Wood & Chan (1994), Dietrich & Newsam (1997)
    
    # Step 1: Compute autocovariance sequence γ(k) for fGn
    gamma = np.zeros(L)
    gamma[0] = 1.0  # γ(0) = variance = 1
    
    for k in range(1, L):
        gamma[k] = 0.5 * ((k + 1)**(2*H) - 2*k**(2*H) + (k - 1)**(2*H))
    
    # Step 2: Construct circulant embedding matrix
    m = 2 * L
    c = np.zeros(m)
    c[:L] = gamma
    c[L] = 0
    c[L+1:] = gamma[L-1:0:-1]
    
    # Step 3: Compute eigenvalues with FFT
    eigenvals = np.real(scipy_fft(c))
    
    # Quick check - if Davies-Harte fails, fall back to Cholesky
    if np.any(eigenvals < -1e-12):
        return _fgn_simple(H, L)

    eigenvals = np.maximum(eigenvals, 0)
    
    # Step 4: Generate Hermitian symmetric complex Gaussian noise
    u = np.random.randn(m)
    v = np.random.randn(m)
    Z = np.zeros(m, dtype=complex)
    Z[0] = u[0]
    
    if m % 2 == 0:
        Z[m // 2] = u[m // 2]
        for k in range(1, m // 2):
            Z[k] = (u[k] + 1j * v[k]) / np.sqrt(2)
            Z[m - k] = (u[k] - 1j * v[k]) / np.sqrt(2)
    else:
        for k in range(1, (m + 1) // 2):
            Z[k] = (u[k] + 1j * v[k]) / np.sqrt(2)
            Z[m - k] = (u[k] - 1j * v[k]) / np.sqrt(2)
    
    # Step 5: Scale by square root of eigenvalues
    W = Z * np.sqrt(eigenvals)
    
    # Step 6: Inverse FFT to get the result
    X = scipy_ifft(W)
    fgn_series = np.real(X[:L])
    
    # Step 7: Normalize to ensure unit variance
    actual_var = np.var(fgn_series)
    if actual_var > 1e-12:
        fgn_series = fgn_series / np.sqrt(actual_var)
    
    return fgn_series


def _fgn_simple(H, L):
    """Fallback simple fGn generation method."""
    # Simple method using covariance matrix (slower but more stable)
    n = np.arange(L)
    i, j = np.meshgrid(n, n, indexing='ij')
    C = 0.5 * (np.abs(i + 1)**(2*H) + np.abs(j + 1)**(2*H) - np.abs(i - j)**(2*H))
    
    # Cholesky decomposition
    try:
        L_chol = np.linalg.cholesky(C)
        Z = np.random.randn(L)
        return L_chol @ Z
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(C)
        eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive
        Z = np.random.randn(L)
        return eigenvecs @ (np.sqrt(eigenvals) * Z)


def fbm(data):
    """
    Compute fractional Brownian motion (fBm) from fractional Gaussian noise (fGn).
    
    Parameters:
        data (np.ndarray): 1D array of values (typically from fgn)
        
    Returns:
        np.ndarray: Array of shape (len(data) + 1,) representing fractional Brownian motion
        
    Raises:
        TypeError: If input is not 1D
    """
    data = np.asarray(data)
    
    if data.ndim != 1:
        raise TypeError(f"Input must be 1D array, got {data.ndim}D")
    
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Input must be numeric array")
    
    # start at 0
    return np.cumsum(np.concatenate([[0], data]))


def fft(data):
    """
    Compute Fast Fourier Transform (FFT) of a 1D signal.
    
    Parameters:
        data (np.ndarray): Real-valued 1D input signal
        
    Returns:
        tuple: (freqs, magnitudes) - frequency bins and FFT magnitudes
    """
    data = np.asarray(data)
    
    if data.ndim != 1:
        raise ValueError(f"Input must be 1D array, got {data.ndim}D")
    
    # Compute FFT
    fft_vals = scipy_fft(data)
    
    # Compute frequency bins (normalized to sampling rate = 1)
    freqs = fftfreq(len(data))
    
    # Compute magnitudes
    magnitudes = np.abs(fft_vals)
    
    return freqs, magnitudes


def fwt(data, wavelet='db2', level=None):
    """
    Compute Fast Wavelet Transform (FWT) of a 1D signal.
    
    Parameters:
        data (np.ndarray): Input signal to transform
        wavelet (str): Wavelet type (default: 'db2')
        level (int or None): Decomposition level; None determines automatically
        
    Returns:
        list: List of np.ndarray objects containing approximation and detail coefficients
        
    Raises:
        ValueError: If wavelet name is invalid
    """
    data = np.asarray(data)
    
    if data.ndim != 1:
        raise ValueError(f"Input must be 1D array, got {data.ndim}D")
    
    # Validate wavelet name
    if wavelet not in pywt.wavelist():
        raise ValueError(f"Invalid wavelet '{wavelet}'. Available wavelets: {pywt.wavelist()}")
    
    # Determine decomposition level if not specified
    if level is None:
        level = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet))
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Verify reconstruction works (as per constraints)
    reconstructed = pywt.waverec(coeffs, wavelet)
    
    # Check if reconstruction is close to original
    if len(reconstructed) != len(data):
        # Handle length mismatch due to padding
        min_len = min(len(reconstructed), len(data))
        reconstructed = reconstructed[:min_len]
        data_check = data[:min_len]
    else:
        data_check = data
    
    if not np.allclose(reconstructed, data_check, rtol=1e-10, atol=1e-12):
        import warnings
        warnings.warn("Wavelet reconstruction differs from original signal", UserWarning)
    
    return coeffs 