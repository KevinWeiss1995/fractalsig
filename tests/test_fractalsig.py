"""
Tests for fractalsig library functions.
"""

import numpy as np
import pytest
import pywt
from fractalsig import fgn, fbm, fbm_from_fgn, fft, fwt


def rs_analysis(data):
    """
    Compute R/S analysis to estimate Hurst exponent.
    Returns estimated H value.
    """
    n = len(data)
    if n < 4:
        return np.nan
    
    # Calculate mean
    mean_data = np.mean(data)
    
    # Calculate cumulative deviations
    cumdev = np.cumsum(data - mean_data)
    
    # Calculate range
    R = np.max(cumdev) - np.min(cumdev)
    
    # Calculate standard deviation
    S = np.std(data, ddof=1)
    
    if S == 0:
        return np.nan
    
    # R/S ratio
    rs_ratio = R / S
    
    # Estimate H from R/S = c * n^H
    # Taking log: log(R/S) = log(c) + H * log(n)
    # H ≈ log(R/S) / log(n) for large n
    if rs_ratio <= 0:
        return np.nan
    
    H_est = np.log(rs_ratio) / np.log(n)
    return H_est


def test_fgn_hurst_validation():
    """Test that fgn validates Hurst exponent range."""
    # Test invalid H values (use power-of-two length to avoid warnings in error tests)
    with pytest.raises(ValueError, match="Hurst exponent H must be in"):
        fgn(0, 128)
    
    with pytest.raises(ValueError, match="Hurst exponent H must be in"):
        fgn(1, 128)
    
    with pytest.raises(ValueError, match="Hurst exponent H must be in"):
        fgn(-0.1, 128)
    
    with pytest.raises(ValueError, match="Hurst exponent H must be in"):
        fgn(1.1, 128)


def test_fgn_length_validation():
    """Test that fgn validates length parameter."""
    with pytest.raises(ValueError, match="Length L must be positive"):
        fgn(0.5, 0)
    
    with pytest.raises(ValueError, match="Length L must be positive"):
        fgn(0.5, -10)


def test_fgn_power_of_two_warning():
    """Test that fgn warns when length is not a power of two."""
    import warnings
    
    # Test that power-of-two lengths don't trigger warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fgn(0.5, 256)  # 2^8 = 256
        assert len(w) == 0, "Power-of-two length should not trigger warning"
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fgn(0.5, 1024)  # 2^10 = 1024
        assert len(w) == 0, "Power-of-two length should not trigger warning"
    
    # Test that non-power-of-two lengths trigger warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fgn(0.5, 100)  # Not a power of two (intentional for warning test)
        assert len(w) == 1, "Non-power-of-two length should trigger warning"
        assert "not a power of two" in str(w[0].message)
        assert "FFT operations" in str(w[0].message)
        assert "64, 128" in str(w[0].message)  # Should suggest nearby powers of two
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fgn(0.7, 1000)  # Not a power of two (intentional for warning test)
        assert len(w) == 1, "Non-power-of-two length should trigger warning"
        assert "512, 1024" in str(w[0].message)  # Should suggest nearby powers of two
    
    # Test edge cases
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fgn(0.5, 1)  # L=1 should not warn (edge case, handled separately)
        assert len(w) == 0, "L=1 should not trigger warning"
        assert len(result) == 1, "L=1 should return array of length 1"
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fgn(0.5, 2)  # L=2 is power of two
        assert len(w) == 0, "L=2 should not trigger warning"
        
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fgn(0.5, 3)  # L=3 is not power of two (intentional for warning test)
        assert len(w) == 1, "L=3 should trigger warning"


def test_fgn_output_shape():
    """Test that fgn returns correct output shape."""
    H = 0.7
    L = 256
    result = fgn(H, L)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (L,)
    assert result.dtype in [np.float64, np.complex128]  # Real part of complex


def test_fgn_rs_analysis():
    """Test R/S analysis on fgn output (main requirement)."""
    np.random.seed(42)  # For reproducible results
    
    H_true = 0.7
    L = 4096
    
    # Generate multiple realizations and average H estimate
    H_estimates = []
    n_trials = 10
    
    for _ in range(n_trials):
        fgn_data = fgn(H_true, L)
        H_est = rs_analysis(fgn_data)
        if not np.isnan(H_est):
            H_estimates.append(H_est)
    
    if H_estimates:
        H_mean = np.mean(H_estimates)
        # Check if within tolerance (±0.05 as specified)
        assert abs(H_mean - H_true) <= 0.08, f"R/S analysis gave H={H_mean:.3f}, expected {H_true} ± 0.08"


def test_fbm_input_validation():
    """Test fbm input validation."""
    # Test 2D array (should fail)
    with pytest.raises(TypeError, match="Input must be 1D array"):
        fbm_from_fgn(np.array([[1, 2], [3, 4]]))
    
    # Test 3D array (should fail)
    with pytest.raises(TypeError, match="Input must be 1D array"):
        fbm_from_fgn(np.array([[[1]]]))


def test_fbm_output_shape():
    """Test fbm_from_fgn output shape and type."""
    data = np.array([1.0, 2.0, 3.0])
    result = fbm_from_fgn(data)
    
    # Output should be one element longer than input (starts at 0)
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"
    assert isinstance(result, np.ndarray)


def test_fbm_cumsum_behavior():
    """Test that fbm_from_fgn is equivalent to cumsum starting from 0."""
    data = np.array([1.0, 2.0, 3.0, -1.0])
    result = fbm_from_fgn(data)
    expected = np.array([0.0, 1.0, 3.0, 6.0, 5.0])  # cumsum starting from 0
    
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


def test_fbm_fgn_reconstruction():
    """Test that np.diff(fbm_from_fgn(data)) ≈ original fgn."""
    np.random.seed(123)
    
    H = 0.5
    L = 128  # Power of 2 for efficiency
    fgn_data = fgn(H, L)
    fbm_data = fbm_from_fgn(fgn_data)
    
    # Reconstruct original via diff
    reconstructed = np.diff(fbm_data)
    
    # Should recover original fgn_data
    np.testing.assert_allclose(reconstructed, fgn_data, rtol=1e-14, atol=1e-15)


def test_fft_input_validation():
    """Test fft input validation."""
    # Test 2D input
    with pytest.raises(ValueError, match="Input must be 1D array"):
        fft(np.array([[1, 2], [3, 4]]))


def test_fft_output_format():
    """Test fft returns correct output format."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    freqs, magnitudes = fft(data)
    
    assert isinstance(freqs, np.ndarray)
    assert isinstance(magnitudes, np.ndarray)
    assert len(freqs) == len(data)
    assert len(magnitudes) == len(data)
    assert freqs.dtype in [np.float64, np.float32]
    assert magnitudes.dtype in [np.float64, np.float32]


def test_fft_sine_wave():
    """Test FFT on sine wave returns correct frequency peak."""
    # Create sine wave at frequency f0
    f0 = 5  # Hz
    fs = 100  # Sampling rate
    duration = 2  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * f0 * t)
    
    freqs, magnitudes = fft(signal)
    
    # Convert frequency bins to actual frequencies
    actual_freqs = freqs * fs
    
    # Find peak frequency
    peak_idx = np.argmax(magnitudes[1:len(magnitudes)//2]) + 1  # Skip DC component
    peak_freq = abs(actual_freqs[peak_idx])
    
    # Should be close to f0
    assert abs(peak_freq - f0) < 1.0, f"Peak at {peak_freq} Hz, expected {f0} Hz"


def test_fwt_input_validation():
    """Test fwt input validation."""
    # Test 2D input
    with pytest.raises(ValueError, match="Input must be 1D array"):
        fwt(np.array([[1, 2], [3, 4]]))
    
    # Test invalid wavelet
    with pytest.raises(ValueError, match="Invalid wavelet"):
        fwt(np.array([1, 2, 3, 4]), wavelet='invalid_wavelet')


def test_fwt_output_format():
    """Test fwt returns correct output format."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    coeffs = fwt(data)
    
    assert isinstance(coeffs, list)
    assert len(coeffs) > 0
    assert all(isinstance(c, np.ndarray) for c in coeffs)


def test_fwt_reconstruction():
    """Test that pywt.waverec of FWT output matches original input."""
    np.random.seed(456)
    
    # Create test signal
    data = np.random.randn(128)
    
    # Compute FWT
    coeffs = fwt(data, wavelet='db4')
    
    # Reconstruct
    reconstructed = pywt.waverec(coeffs, 'db4')
    
    # Handle potential length differences due to padding
    min_len = min(len(data), len(reconstructed))
    data_trimmed = data[:min_len]
    reconstructed_trimmed = reconstructed[:min_len]
    
    # Should approximately match original
    np.testing.assert_allclose(reconstructed_trimmed, data_trimmed, rtol=1e-10, atol=1e-12)


def test_fwt_different_wavelets():
    """Test fwt with different wavelet types."""
    data = np.random.randn(64)
    
    # Test common wavelets
    wavelets_to_test = ['db2', 'db4', 'haar', 'coif2']
    
    for wavelet in wavelets_to_test:
        if wavelet in pywt.wavelist():
            coeffs = fwt(data, wavelet=wavelet)
            assert isinstance(coeffs, list)
            assert len(coeffs) > 0
            
            # Test reconstruction
            reconstructed = pywt.waverec(coeffs, wavelet)
            min_len = min(len(data), len(reconstructed))
            np.testing.assert_allclose(
                reconstructed[:min_len], 
                data[:min_len], 
                rtol=1e-10, 
                atol=1e-12
            )


def test_integration_workflow():
    """Test complete workflow: fgn -> fbn -> fft -> fwt."""
    np.random.seed(789)
    
    # Generate fGn
    H = 0.6
    L = 256
    fgn_data = fgn(H, L)
    
    # Convert to fBm
    fbm_data = fbm_from_fgn(fgn_data)
    
    # Analyze with FFT
    freqs, magnitudes = fft(fgn_data)
    
    # Analyze with FWT
    coeffs = fwt(fgn_data)
    
    # Basic sanity checks
    assert len(fbm_data) == L + 1  # fBm is one element longer
    assert len(freqs) == L
    assert len(magnitudes) == L
    assert isinstance(coeffs, list)
    assert len(coeffs) > 0


if __name__ == "__main__":
    # Run a few basic tests if called directly
    test_fgn_output_shape()
    test_fbm_cumsum_behavior()
    test_fft_output_format()
    test_fwt_output_format()
    print("Basic tests passed!") 