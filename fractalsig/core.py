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
    
    # Reference: Percival & Walden (2000), Wavelet Methods for Time Series Analysis
    
    # Step 1: Create the autocovariance sequence γ(k) for fGn
    # γ(k) = 0.5 * [|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H)]
    k = np.arange(L, dtype=float)
    gamma = np.zeros(L)
    
    for i in range(L):
        if i == 0:
            gamma[i] = 1.0  # γ(0) = Var = 1
        else:
            gamma[i] = 0.5 * (abs(i + 1)**(2*H) - 2*abs(i)**(2*H) + abs(i - 1)**(2*H))
    
    # Step 2: Create circulant embedding matrix first row
    # Need size m = 2(n-1) to ensure positive semi-definite
    m = 2 * (L - 1)
    c = np.zeros(m)
    
    # Fill first row of circulant matrix:
    # c = [γ(0), γ(1), ..., γ(n-1), γ(n-1), γ(n-2), ..., γ(1)]
    c[0:L] = gamma  # γ(0) to γ(n-1)
    c[L:m] = gamma[L-2:0:-1]  # γ(n-2) down to γ(1), reverse order
    
    # Step 3: Get eigenvalues of circulant matrix via FFT
    eigenvals = np.real(scipy_fft(c))
    
    # Check for negative eigenvalues (method failure)
    if np.any(eigenvals < -1e-14):
        return _fgn_simple(H, L)
    
    # Ensure all eigenvalues are non-negative
    eigenvals = np.maximum(eigenvals, 0)
    
    # Step 4: Generate independent standard Gaussian random variables
    # For real-valued result, we need proper conjugate symmetry
    
    # Method from Dietrich & Newsam (1997) - the definitive reference
    a = np.random.randn(m)
    b = np.random.randn(m)
    
    # Create complex white noise with proper symmetry
    V = np.zeros(m, dtype=complex)
    
    # V[0] is real (DC component)
    V[0] = a[0]
    
    # V[m/2] is real if m is even (Nyquist component)
    if m % 2 == 0:
        V[m // 2] = a[m // 2]
        # Fill symmetric pairs for k = 1, ..., m/2-1
        for k in range(1, m // 2):
            V[k] = a[k] + 1j * b[k]
            V[m - k] = a[k] - 1j * b[k]  # Conjugate symmetry
    else:
        # m is odd, fill symmetric pairs for k = 1, ..., (m-1)/2
        for k in range(1, (m + 1) // 2):
            V[k] = a[k] + 1j * b[k]
            V[m - k] = a[k] - 1j * b[k]  # Conjugate symmetry
    
    # Step 5: Scale by square root of eigenvalues
    W = V * np.sqrt(eigenvals)
    
    # Step 6: Inverse FFT to get the fGn realization
    X = scipy_ifft(W)
    
    # Take real part and first L values
    fgn_series = np.real(X[:L])
    
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