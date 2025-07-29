"""
Core functions for fractional signal analysis.
"""

import numpy as np
import pywt
from scipy.fft import fft as scipy_fft, ifft as scipy_ifft, fftfreq


def fgn(H, L):
    """
    Generate a fractional Gaussian noise (fGn) time series using Davies-Harte method.
    
    Parameters:
        H (float): Hurst exponent (0 < H < 1)
        L (int): Length of the time series
        
    Returns:
        np.ndarray: Array of shape (L,) representing fractional Gaussian noise
        
    Raises:
        ValueError: If H is not in (0, 1)
        UserWarning: If L is not a power of two
    """
    if not (0 < H < 1):
        raise ValueError(f"Hurst exponent H must be in (0, 1), got {H}")
    
    if L <= 0:
        raise ValueError(f"Length L must be positive, got {L}")
    
    # Handle edge case for very small lengths
    if L == 1:
        return np.array([np.random.randn()])
    
    # Check if L is a power of two and warn if not
    if L > 1 and (L & (L - 1)) != 0:
        import warnings
        # Find the next power of two for suggestion
        next_power_of_two = 1 << (L - 1).bit_length()
        prev_power_of_two = next_power_of_two >> 1
        
        warnings.warn(
            f"Length L={L} is not a power of two. The Davies-Harte method uses FFT operations "
            f"which are most efficient with power-of-two lengths (e.g., {prev_power_of_two}, {next_power_of_two}). "
            f"Non-power-of-two lengths will result in slower FFT computations and may cause "
            f"the circulant embedding matrix to have a larger size (2*(L-1)={2*(L-1)}), "
            f"potentially leading to increased memory usage and computation time. "
            f"Consider using a nearby power of two for optimal performance.",
            UserWarning,
            stacklevel=2
        )
    
    # Davies-Harte method implementation
    # Create autocovariance function for fGn
    n = np.arange(L)
    r = 0.5 * ((n + 1)**(2*H) - 2*n**(2*H) + np.abs(n - 1)**(2*H))
    r[0] = 1.0  # Variance at lag 0
    
    # Extend for circulant embedding
    N = 2 * (L - 1)
    R = np.zeros(N)
    R[:L] = r
    R[L:] = r[1:L-1][::-1]  # Mirror for circulant structure
    
    # Eigenvalues via FFT
    lambda_vals = np.real(scipy_fft(R))
    
    # Check for numerical issues
    if np.any(lambda_vals < -1e-12):
        # Fall back to simple method if circulant embedding fails
        return _fgn_simple(H, L)
    
    # Ensure non-negative eigenvalues
    lambda_vals = np.maximum(lambda_vals, 0)
    
    # Generate random Gaussian variables
    W = np.random.randn(N) + 1j * np.random.randn(N)
    W[0] = np.real(W[0]) * np.sqrt(2)  # DC component is real
    W[L-1] = np.real(W[L-1]) * np.sqrt(2)  # Nyquist frequency is real
    
    # Apply square root of eigenvalues
    Y = W * np.sqrt(lambda_vals)
    
    # Inverse FFT and take real part
    fgn_series = np.real(scipy_ifft(Y))[:L]
    
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