"""
Analysis utilities for FractalSig library.
Provides statistical analysis and parameter estimation functions.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union
import warnings
from scipy.fft import fft, fftfreq


def rs_analysis(data: np.ndarray, max_lag: Optional[int] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Perform Rescaled Range (R/S) analysis to estimate Hurst exponent.
    
    Parameters:
        data: Time series data
        max_lag: Maximum lag for analysis (default: len(data)//4)
        
    Returns:
        Tuple of (estimated_H, window_sizes, rs_values)
    """
    if max_lag is None:
        max_lag = len(data) // 4
    
    # Generate window sizes (logarithmic spacing)
    min_window = 8
    max_window = min(max_lag, len(data) // 2)
    
    if max_window < min_window:
        raise ValueError(f"Data too short for R/S analysis. Need at least {min_window*2} points")
    
    n_windows = 15
    window_sizes = np.logspace(np.log10(min_window), np.log10(max_window), n_windows).astype(int)
    window_sizes = np.unique(window_sizes)
    
    rs_values = []
    
    for window_size in window_sizes:
        if window_size >= len(data):
            continue
            
        # Number of non-overlapping windows
        n_segments = len(data) // window_size
        rs_segment_values = []
        
        for i in range(n_segments):
            segment = data[i * window_size:(i + 1) * window_size]
            
            # Calculate mean
            mean_segment = np.mean(segment)
            
            # Calculate cumulative departures from mean
            cumulative_departures = np.cumsum(segment - mean_segment)
            
            # Calculate range
            R = np.max(cumulative_departures) - np.min(cumulative_departures)
            
            # Calculate standard deviation
            S = np.std(segment, ddof=1)
            
            # Calculate R/S ratio
            if S > 0:
                rs_segment_values.append(R / S)
        
        if rs_segment_values:
            rs_values.append(np.mean(rs_segment_values))
    
    rs_values = np.array(rs_values)
    valid_window_sizes = window_sizes[:len(rs_values)]
    
    if len(rs_values) < 3:
        raise ValueError("Not enough valid R/S values for Hurst estimation")
    
    # Estimate Hurst exponent using linear regression in log-log space
    log_windows = np.log10(valid_window_sizes)
    log_rs = np.log10(rs_values)
    
    # Linear fit: log(R/S) = log(c) + H * log(n)
    H_estimate = np.polyfit(log_windows, log_rs, 1)[0]
    
    return H_estimate, valid_window_sizes, rs_values


def dfa_analysis(data: np.ndarray, min_window: int = 8, max_window: Optional[int] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Perform Detrended Fluctuation Analysis (DFA) to estimate Hurst exponent.
    
    Parameters:
        data: Time series data
        min_window: Minimum window size
        max_window: Maximum window size (default: len(data)//4)
        
    Returns:
        Tuple of (estimated_H, window_sizes, fluctuation_values)
    """
    if max_window is None:
        max_window = len(data) // 4
    
    # Integrate the data (cumulative sum after removing mean)
    y = np.cumsum(data - np.mean(data))
    
    # Window sizes (logarithmic spacing)
    n_windows = 15
    window_sizes = np.logspace(np.log10(min_window), np.log10(max_window), n_windows).astype(int)
    window_sizes = np.unique(window_sizes)
    
    fluctuations = []
    
    for window_size in window_sizes:
        if window_size >= len(y):
            continue
            
        # Number of non-overlapping windows
        n_segments = len(y) // window_size
        
        segment_fluctuations = []
        
        for i in range(n_segments):
            segment = y[i * window_size:(i + 1) * window_size]
            
            # Fit linear trend
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            
            # Calculate fluctuation
            fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
            segment_fluctuations.append(fluctuation)
        
        if segment_fluctuations:
            fluctuations.append(np.mean(segment_fluctuations))
    
    fluctuations = np.array(fluctuations)
    valid_window_sizes = window_sizes[:len(fluctuations)]
    
    if len(fluctuations) < 3:
        raise ValueError("Not enough valid fluctuation values for DFA")
    
    # Estimate scaling exponent using linear regression in log-log space
    log_windows = np.log10(valid_window_sizes)
    log_fluctuations = np.log10(fluctuations)
    
    # Linear fit: log(F) = log(c) + α * log(n), where H = α for fGn
    H_estimate = np.polyfit(log_windows, log_fluctuations, 1)[0]
    
    return H_estimate, valid_window_sizes, fluctuations


def wavelet_hurst_estimation(data: np.ndarray, wavelet: str = 'db4') -> Tuple[float, Dict]:
    """
    Estimate Hurst exponent using wavelet-based method.
    
    Parameters:
        data: Time series data
        wavelet: Wavelet type to use
        
    Returns:
        Tuple of (estimated_H, analysis_info)
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("PyWavelets is required for wavelet-based Hurst estimation")
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, mode='periodization')
    
    # Calculate energy at each scale
    energies = []
    scales = []
    
    for j, detail_coeffs in enumerate(coeffs[1:], 1):  # Skip approximation coefficients
        if len(detail_coeffs) > 0:
            energy = np.mean(detail_coeffs ** 2)
            energies.append(energy)
            scales.append(2 ** j)
    
    if len(energies) < 3:
        raise ValueError("Not enough wavelet scales for Hurst estimation")
    
    energies = np.array(energies)
    scales = np.array(scales)
    
    # Remove zero energies (can cause issues in log space)
    valid_mask = energies > 0
    energies = energies[valid_mask]
    scales = scales[valid_mask]
    
    if len(energies) < 3:
        raise ValueError("Not enough non-zero energies for Hurst estimation")
    
    # Linear regression in log-log space: log(E_j) = log(c) + (2H+1) * log(2^j)
    log_scales = np.log2(scales)
    log_energies = np.log2(energies)
    
    slope = np.polyfit(log_scales, log_energies, 1)[0]
    H_estimate = (slope - 1) / 2
    
    analysis_info = {
        'scales': scales,
        'energies': energies,
        'slope': slope,
        'wavelet': wavelet,
        'n_scales': len(scales)
    }
    
    return H_estimate, analysis_info


def estimate_hurst_multiple_methods(data: np.ndarray) -> Dict[str, Union[float, Dict]]:
    """
    Estimate Hurst exponent using multiple methods and return comparison.
    
    Parameters:
        data: Time series data
        
    Returns:
        Dictionary with results from different methods
    """
    results = {}
    
    # R/S Analysis
    try:
        h_rs, _, _ = rs_analysis(data)
        results['rs_analysis'] = {'H': h_rs, 'method': 'Rescaled Range'}
    except Exception as e:
        results['rs_analysis'] = {'error': str(e)}
    
    # DFA
    try:
        h_dfa, _, _ = dfa_analysis(data)
        results['dfa'] = {'H': h_dfa, 'method': 'Detrended Fluctuation Analysis'}
    except Exception as e:
        results['dfa'] = {'error': str(e)}
    
    # Wavelet method
    try:
        h_wavelet, wavelet_info = wavelet_hurst_estimation(data)
        results['wavelet'] = {'H': h_wavelet, 'method': 'Wavelet-based', 'info': wavelet_info}
    except Exception as e:
        results['wavelet'] = {'error': str(e)}
    
    # Calculate statistics if we have multiple successful estimates
    successful_estimates = []
    for method, result in results.items():
        if 'H' in result:
            successful_estimates.append(result['H'])
    
    if len(successful_estimates) > 1:
        results['summary'] = {
            'mean_H': np.mean(successful_estimates),
            'std_H': np.std(successful_estimates),
            'min_H': np.min(successful_estimates),
            'max_H': np.max(successful_estimates),
            'n_methods': len(successful_estimates)
        }
    
    return results


def autocorrelation_function(data: np.ndarray, max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate autocorrelation function of time series.
    
    Parameters:
        data: Time series data
        max_lag: Maximum lag to compute (default: len(data)//4)
        
    Returns:
        Tuple of (lags, autocorrelations)
    """
    if max_lag is None:
        max_lag = len(data) // 4
    
    # Normalize data
    data_normalized = data - np.mean(data)
    
    # Compute full autocorrelation
    autocorr_full = np.correlate(data_normalized, data_normalized, mode='full')
    
    # Take positive lags only
    mid = len(autocorr_full) // 2
    autocorr = autocorr_full[mid:mid + max_lag + 1]
    
    # Normalize by variance (lag 0)
    autocorr = autocorr / autocorr[0]
    
    lags = np.arange(max_lag + 1)
    
    return lags, autocorr


def power_spectral_density(data: np.ndarray, method: str = 'periodogram') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectral density of time series.
    
    Parameters:
        data: Time series data
        method: Method to use ('periodogram', 'welch')
        
    Returns:
        Tuple of (frequencies, psd)
    """
    try:
        from scipy import signal
    except ImportError:
        # Fallback to simple periodogram using FFT
        method = 'fft'
    
    if method == 'periodogram' and 'signal' in locals():
        freqs, psd = signal.periodogram(data)
    elif method == 'welch' and 'signal' in locals():
        freqs, psd = signal.welch(data)
    else:
        # Simple FFT-based PSD
        fft_data = fft(data)
        psd = np.abs(fft_data) ** 2 / len(data)
        freqs = fftfreq(len(data))
        
        # Take positive frequencies only
        positive_mask = freqs >= 0
        freqs = freqs[positive_mask]
        psd = psd[positive_mask]
    
    return freqs, psd


def validate_fgn_properties(data: np.ndarray, H: float, alpha: float = 0.05) -> Dict[str, Union[bool, float, str]]:
    """
    Validate statistical properties of fractional Gaussian noise.
    
    Parameters:
        data: fGn time series
        H: Expected Hurst exponent
        alpha: Significance level for tests
        
    Returns:
        Dictionary with validation results
    """
    results = {}
    n = len(data)
    
    # 1. Normality test (Shapiro-Wilk if available, otherwise simple moment tests)
    try:
        from scipy import stats
        if n <= 5000:  # Shapiro-Wilk works well for smaller samples
            stat, p_value = stats.shapiro(data)
            results['normality'] = {
                'test': 'Shapiro-Wilk',
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > alpha
            }
        else:
            # Use Kolmogorov-Smirnov test for larger samples
            stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            results['normality'] = {
                'test': 'Kolmogorov-Smirnov',
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > alpha
            }
    except ImportError:
        # Fallback: simple moment-based check
        skewness = np.mean(((data - np.mean(data)) / np.std(data)) ** 3)
        kurtosis = np.mean(((data - np.mean(data)) / np.std(data)) ** 4) - 3
        
        results['normality'] = {
            'test': 'Moment-based',
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': abs(skewness) < 0.5 and abs(kurtosis) < 1.0  # Rough criteria
        }
    
    # 2. Stationarity check (mean and variance)
    # Split data into segments and check for consistency
    n_segments = 5
    segment_size = n // n_segments
    
    if segment_size > 10:
        segment_means = []
        segment_vars = []
        
        for i in range(n_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size, n)
            segment = data[start:end]
            segment_means.append(np.mean(segment))
            segment_vars.append(np.var(segment))
        
        # Check if means are consistent (should be close to 0)
        mean_consistency = np.std(segment_means) < 0.1 * np.std(data)
        var_consistency = np.std(segment_vars) < 0.2 * np.var(data)
        
        results['stationarity'] = {
            'mean_consistency': mean_consistency,
            'variance_consistency': var_consistency,
            'is_stationary': mean_consistency and var_consistency,
            'segment_means': segment_means,
            'segment_variances': segment_vars
        }
    
    # 3. Hurst exponent validation
    try:
        H_estimated, _, _ = rs_analysis(data)
        H_error = abs(H_estimated - H)
        
        results['hurst_validation'] = {
            'expected_H': H,
            'estimated_H': H_estimated,
            'error': H_error,
            'is_accurate': H_error < 0.1,  # Within 0.1 tolerance
            'method': 'R/S Analysis'
        }
    except Exception as e:
        results['hurst_validation'] = {'error': str(e)}
    
    # 4. Autocorrelation validation
    lags, autocorr = autocorrelation_function(data, max_lag=min(50, n//4))
    
    # For fGn, autocorrelation should decay as a power law
    if len(lags) > 5:
        # Check if autocorrelation is positive for H > 0.5 (long-range dependence)
        if H > 0.5:
            positive_correlation = np.mean(autocorr[1:11]) > 0  # First 10 lags
        else:
            positive_correlation = True  # No specific requirement for H <= 0.5
        
        results['autocorrelation'] = {
            'has_long_range_dependence': positive_correlation if H > 0.5 else None,
            'first_lag_correlation': autocorr[1] if len(autocorr) > 1 else None,
            'mean_correlation_1_10': np.mean(autocorr[1:11]) if len(autocorr) > 10 else None
        }
    
    return results


def generate_confidence_intervals(data: np.ndarray, H: float, confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Generate confidence intervals for fGn properties.
    
    Parameters:
        data: fGn time series
        H: Hurst exponent
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Dictionary with confidence intervals
    """
    n = len(data)
    z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
    
    results = {}
    
    # Mean confidence interval
    mean_val = np.mean(data)
    std_val = np.std(data)
    mean_se = std_val / np.sqrt(n)
    results['mean'] = (mean_val - z_score * mean_se, mean_val + z_score * mean_se)
    
    # Variance confidence interval (chi-square distribution)
    try:
        from scipy import stats
        chi2_lower = stats.chi2.ppf((1 - confidence_level) / 2, n - 1)
        chi2_upper = stats.chi2.ppf((1 + confidence_level) / 2, n - 1)
        
        var_val = np.var(data, ddof=1)
        var_lower = (n - 1) * var_val / chi2_upper
        var_upper = (n - 1) * var_val / chi2_lower
        results['variance'] = (var_lower, var_upper)
    except ImportError:
        # Approximate confidence interval for variance
        var_val = np.var(data, ddof=1)
        var_se = var_val * np.sqrt(2 / (n - 1))  # Approximate standard error
        results['variance'] = (var_val - z_score * var_se, var_val + z_score * var_se)
    
    # Hurst exponent confidence interval (approximate)
    try:
        H_estimated, _, _ = rs_analysis(data)
        # Approximate standard error for Hurst exponent (empirical formula)
        H_se = 0.05 + 0.02 / np.sqrt(np.log10(n))  # Rough approximation
        results['hurst'] = (H_estimated - z_score * H_se, H_estimated + z_score * H_se)
    except:
        results['hurst'] = None
    
    return results


def compare_time_series(data1: np.ndarray, data2: np.ndarray, 
                       labels: Optional[Tuple[str, str]] = None) -> Dict[str, Union[float, bool]]:
    """
    Compare two time series and return statistical comparison metrics.
    
    Parameters:
        data1: First time series
        data2: Second time series
        labels: Optional labels for the series
        
    Returns:
        Dictionary with comparison metrics
    """
    if labels is None:
        labels = ('Series 1', 'Series 2')
    
    results = {'series_1_label': labels[0], 'series_2_label': labels[1]}
    
    # Basic statistics comparison
    results['mean_difference'] = abs(np.mean(data1) - np.mean(data2))
    results['std_ratio'] = np.std(data1) / np.std(data2)
    results['length_ratio'] = len(data1) / len(data2)
    
    # Correlation (if same length)
    min_len = min(len(data1), len(data2))
    if min_len > 1:
        data1_trunc = data1[:min_len]
        data2_trunc = data2[:min_len]
        correlation = np.corrcoef(data1_trunc, data2_trunc)[0, 1]
        results['correlation'] = correlation
    
    # Hurst exponent comparison
    try:
        H1, _, _ = rs_analysis(data1)
        H2, _, _ = rs_analysis(data2)
        results['hurst_1'] = H1
        results['hurst_2'] = H2
        results['hurst_difference'] = abs(H1 - H2)
    except:
        results['hurst_comparison'] = 'Failed'
    
    # Distribution comparison (approximate)
    # Using overlapping coefficient
    hist1, bins = np.histogram(data1, bins=50, density=True)
    hist2, _ = np.histogram(data2, bins=bins, density=True)
    overlap = np.sum(np.minimum(hist1, hist2)) * (bins[1] - bins[0])
    results['distribution_overlap'] = overlap
    
    return results 