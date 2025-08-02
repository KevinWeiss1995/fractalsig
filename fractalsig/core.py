"""
Core functions for fractional signal analysis.
"""

import numpy as np
import pywt
from scipy.fft import fft as scipy_fft, ifft as scipy_ifft, fftfreq


def fgn(H, L, length=1):
    """
    Generate fractional Gaussian noise using circulant embedding method.
    
    Parameters:
        H (float): Hurst exponent (0 < H < 1)
        L (int): Length of the time series
        length (float): Length of realization (default=1)
        
    Returns:
        np.ndarray: Array of shape (L,) representing fractional Gaussian noise
        
    Raises:
        ValueError: If H is not in (0, 1)
    """
    if not (0 < H < 1):
        raise ValueError(f'Hurst exponent H must be in (0, 1), got {H}')
    
    if L <= 0:
        raise ValueError(f'Length L must be positive, got {L}')
    
    # Handle edge case for very small lengths
    if L == 1:
        return np.array([np.random.randn()])
    
    # Check if L is a power of two and warn if not (for FFT efficiency)
    import warnings
    if L > 0 and (L & (L - 1)) != 0:  # Check if not power of two
        # Find nearby powers of two
        lower_power = 1 << (L - 1).bit_length() - 1
        higher_power = 1 << (L - 1).bit_length()
        warnings.warn(
            f"Length {L} is not a power of two. FFT operations are most efficient "
            f"with power-of-two lengths. Consider using {lower_power}, {higher_power} "
            f"or other powers of two for optimal performance.",
            UserWarning
        )

    # 1) Create a time vector
    t = np.linspace(0, length, L+1)

    # 2) Create autocovariance function
    gamma = lambda k: 0.5 * (abs(k+1)**(2*H) - 2*abs(k)**(2*H) + abs(k-1)**(2*H))

    # 3) Create circulant vector of autocovariance
    c = np.concatenate([np.array([gamma(k) for k in range(L+1)]), np.array([gamma(k) for k in range(L-1, 0, -1)])])

    # 4) Compute eigenvalues using FFT
    # Note: we need to consider how FFT/IFFT scales by default in Python! In Python, numpy FFT computes the
    # unnormalized discrete Fourier transform.
    eigenvals = np.fft.fft(c).real
    if not np.allclose(np.fft.fft(c).imag, 0, atol=1e-10):
        raise ValueError("FFT has significant imaginary component, check input vector")

    if np.any(eigenvals < 0):
        raise ValueError("FFT has negative eigenvalues, check circulant embedding")

    # 5) Generate complex Gaussian vector
    # Note: Vector of eigenvalues Lambda is going to be larger than L, length 2L, we truncate 
    # the vector to L as needed
    M = 2*L # FFT length

    Z = np.zeros(M, dtype=np.complex128)

    # Real parts
    Z[0] = np.sqrt(eigenvals[0]) * np.random.normal()
    Z[L] = np.sqrt(eigenvals[L]) * np.random.normal()

    # 1 <= k < L
    X = np.random.normal(0, 1, L-1)
    Y = np.random.normal(0, 1, L-1)

    for k in range(1, L):
        Z[k] = np.sqrt(eigenvals[k] / 2) * (X[k-1] + 1j * Y[k-1])
        Z[M-k] = np.conj(Z[k])

    # 6) Inverse FFT to get fractional Gaussian noise
    # Note: in Python IFFT introduces a factor of 1/M (total path length of 2L) which will mess
    # up the scaling and give incorrect variance of fractional Gaussian noise. We need to scale
    # by sqrt(M). Moreover, the scheme assumes unit variance in the inverse FFT so we also 
    # scale by (length/L)**H to have correct time scaling

    fGn = np.fft.ifft(Z).real[:L] * (length/L) ** H * np.sqrt(M)
    
    return fGn


def fbm(H, L, length=1):
    """
    Generate fractional Brownian motion directly.
    
    Parameters:
        H (float): Hurst exponent (0 < H < 1)
        L (int): Number of increments
        length (float): Length of realization (default=1)
        
    Returns:
        np.ndarray: Array of shape (L+1,) representing fractional Brownian motion
        
    Raises:
        ValueError: If H is not in (0, 1)
    """
    # Generate fGn first
    fgn_data = fgn(H, L, length)
    
    # Convert to fBm: cumulative sum with 0 at beginning
    return np.concatenate([np.array([0]), np.cumsum(fgn_data)])


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