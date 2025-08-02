"""
Utility functions for FractalSig library.
Provides benchmarking, validation, and helper utilities.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import warnings


def benchmark_fgn_methods(H_values: List[float], L_values: List[int], 
                         n_trials: int = 5) -> Dict[str, Any]:
    """
    Benchmark different fGn generation methods across various parameters.
    
    Parameters:
        H_values: List of Hurst exponents to test
        L_values: List of time series lengths to test
        n_trials: Number of trials for each parameter combination
        
    Returns:
        Dictionary with benchmark results
    """
    from .core import fgn  # Import here to avoid circular import
    
    results = {
        'H_values': H_values,
        'L_values': L_values,
        'n_trials': n_trials,
        'timings': {},
        'accuracies': {},
        'memory_usage': {},
        'method_success_rates': {}
    }
    
    for H in H_values:
        results['timings'][H] = {}
        results['accuracies'][H] = {}
        
        for L in L_values:
            trial_times = []
            successful_trials = 0
            hurst_estimates = []
            
            for trial in range(n_trials):
                try:
                    # Time the generation
                    start_time = time.time()
                    data = fgn(H, L)
                    end_time = time.time()
                    
                    trial_times.append(end_time - start_time)
                    successful_trials += 1
                    
                    # Quick Hurst estimation for accuracy check
                    if L >= 64:  # Only for longer series
                        try:
                            from .analysis import rs_analysis
                            H_est, _, _ = rs_analysis(data)
                            hurst_estimates.append(H_est)
                        except:
                            pass
                    
                except Exception as e:
                    warnings.warn(f"Trial failed for H={H}, L={L}: {str(e)}")
            
            # Store results
            if trial_times:
                results['timings'][H][L] = {
                    'mean_time': np.mean(trial_times),
                    'std_time': np.std(trial_times),
                    'min_time': np.min(trial_times),
                    'max_time': np.max(trial_times)
                }
            
            results['method_success_rates'][f'H{H}_L{L}'] = successful_trials / n_trials
            
            if hurst_estimates:
                accuracy = np.mean([abs(h - H) for h in hurst_estimates])
                results['accuracies'][H][L] = {
                    'mean_error': accuracy,
                    'estimated_hurst_values': hurst_estimates
                }
    
    return results


def validate_algorithm_correctness(n_tests: int = 5) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate the correctness of core algorithms with known properties.
    
    Parameters:
        n_tests: Number of random tests to perform
        
    Returns:
        Dictionary with test results and any error messages
    """
    from .core import fgn, fbm, fft, fwt
    from .analysis import rs_analysis
    import pywt
    
    results = {
        'fgn_generation': True,
        'fbm_reconstruction': True,
        'fft_sine_wave': True,
        'wavelet_reconstruction': True,
        'rs_analysis_consistency': True,
        'error_messages': []
    }
    
    try:
        # Test 1: fGn generation with known parameters
        for _ in range(n_tests):
            H = np.random.uniform(0.1, 0.9)
            # Use power-of-two lengths for optimal FFT performance
            power_of_two_lengths = [64, 128, 256, 512]
            L = np.random.choice(power_of_two_lengths)
            
            data = fgn(H, L)
            
            # Basic checks
            if len(data) != L:
                results['fgn_generation'] = False
                results['error_messages'].append(f"fGn length mismatch: expected {L}, got {len(data)}")
            
            if not np.isfinite(data).all():
                results['fgn_generation'] = False
                results['error_messages'].append("fGn contains non-finite values")
        
        # Test 2: fBm generation and structure
        for _ in range(n_tests):
            L = 64
            fbm_data = fbm(0.7, L)
            
            # fBm should have L+1 elements and start at 0
            if len(fbm_data) != L + 1:
                results['fbm_reconstruction'] = False
                results['error_messages'].append(f"fBm length incorrect: expected {L + 1}, got {len(fbm_data)}")
            
            if fbm_data[0] != 0.0:
                results['fbm_reconstruction'] = False
                results['error_messages'].append(f"fBm should start at 0, got {fbm_data[0]}")
            
            # Differences should have proper length
            diffs = np.diff(fbm_data)
            if len(diffs) != L:
                results['fbm_reconstruction'] = False
                results['error_messages'].append(f"fBm differences should have length {L}, got {len(diffs)}")
        
        # Test 3: FFT on known sine wave
        for freq in [5, 10, 20]:
            t = np.linspace(0, 1, 128, endpoint=False)
            sine_wave = np.sin(2 * np.pi * freq * t)
            freqs, mags = fft(sine_wave)
            
            # Find peak frequency
            positive_mask = freqs >= 0
            pos_freqs = freqs[positive_mask] * 128  # Convert to Hz
            pos_mags = mags[positive_mask]
            peak_idx = np.argmax(pos_mags)
            peak_freq = pos_freqs[peak_idx]
            
            if abs(peak_freq - freq) > 1.0:  # Allow 1 Hz tolerance
                results['fft_sine_wave'] = False
                results['error_messages'].append(f"FFT peak detection failed: expected {freq} Hz, got {peak_freq:.1f} Hz")
        
        # Test 4: Wavelet reconstruction
        test_wavelets = ['db2', 'haar', 'bior2.2']
        for wavelet in test_wavelets:
            test_signal = np.random.randn(128)
            try:
                coeffs = fwt(test_signal, wavelet=wavelet)
                reconstructed = pywt.waverec(coeffs, wavelet)
                
                # Handle potential length differences due to padding
                min_len = min(len(test_signal), len(reconstructed))
                if not np.allclose(test_signal[:min_len], reconstructed[:min_len], rtol=1e-10):
                    results['wavelet_reconstruction'] = False
                    results['error_messages'].append(f"Wavelet {wavelet} reconstruction failed")
            except Exception as e:
                results['wavelet_reconstruction'] = False
                results['error_messages'].append(f"Wavelet {wavelet} error: {str(e)}")
        
        # Test 5: R/S analysis consistency
        for _ in range(3):
            H_true = np.random.uniform(0.2, 0.8)
            test_data = fgn(H_true, 256)
            H_estimated, _, _ = rs_analysis(test_data)
            
            # R/S analysis can be noisy, allow generous tolerance
            if abs(H_estimated - H_true) > 0.3:
                results['rs_analysis_consistency'] = False
                results['error_messages'].append(f"R/S analysis: H_true={H_true:.2f}, H_est={H_estimated:.2f}")
                
    except Exception as e:
        results['error_messages'].append(f"Unexpected error in validation: {str(e)}")
        for key in results.keys():
            if key != 'error_messages':
                results[key] = False
    
    # Summary
    all_passed = all([
        results['fgn_generation'],
        results['fbm_reconstruction'],
        results['fft_sine_wave'],
        results['wavelet_reconstruction'],
        results['rs_analysis_consistency']
    ])
    
    return results


def generate_test_dataset(H: float, L: int, noise_level: float = 0.0, 
                         trend: Optional[str] = None, 
                         seasonal_period: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate synthetic test dataset with known properties.
    
    Parameters:
        H: Hurst exponent
        L: Length of time series
        noise_level: Additional white noise level (0-1)
        trend: Type of trend ('linear', 'quadratic', 'exponential')
        seasonal_period: Period for seasonal component
        
    Returns:
        Dictionary with generated data and components
    """
    from .core import fgn, fbm
    
    # Generate base fGn
    base_fgn = fgn(H, L)
    
    result = {
        'fgn': base_fgn,
        'fbm': fbm(H, L),
        'H_true': H,
        'length': L
    }
    
    # Add components
    t = np.arange(L)
    
    # Trend component
    if trend == 'linear':
        trend_component = 0.01 * t
        result['trend'] = trend_component
    elif trend == 'quadratic': 
        trend_component = 0.0001 * t**2
        result['trend'] = trend_component
    elif trend == 'exponential':
        trend_component = 0.001 * np.exp(0.005 * t)
        result['trend'] = trend_component
    else:
        trend_component = np.zeros(L)
        result['trend'] = trend_component
    
    # Seasonal component
    if seasonal_period is not None:
        seasonal_component = 0.5 * np.sin(2 * np.pi * t / seasonal_period)
        result['seasonal'] = seasonal_component
    else:
        seasonal_component = np.zeros(L)
        result['seasonal'] = seasonal_component
    
    # White noise
    if noise_level > 0:
        white_noise = noise_level * np.random.randn(L)
        result['white_noise'] = white_noise
    else:
        white_noise = np.zeros(L)
        result['white_noise'] = white_noise
    
    # Combined signal
    result['combined'] = base_fgn + trend_component + seasonal_component + white_noise
    
    return result


def memory_usage_profile(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Profile memory usage of a function call.
    
    Parameters:
        func: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with memory usage statistics
    """
    import gc
    import psutil
    import os
    
    # Force garbage collection before measurement
    gc.collect()
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Execute function
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        success = True
        error_msg = None
    except Exception as e:
        result = None
        success = False
        error_msg = str(e)
    
    end_time = time.time()
    
    # Force garbage collection after execution
    gc.collect()
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'memory_increase_mb': final_memory - initial_memory,
        'execution_time_seconds': end_time - start_time,
        'success': success,
        'error_message': error_msg,
        'result': result
    }


def export_data(data: Dict[str, np.ndarray], filename: str, format: str = 'npz') -> bool:
    """
    Export data to various formats.
    
    Parameters:
        data: Dictionary of arrays to export
        filename: Output filename
        format: Export format ('npz', 'csv', 'mat')
        
    Returns:
        Success status
    """
    try:
        if format == 'npz':
            np.savez(filename, **data)
        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
        elif format == 'mat':
            try:
                from scipy.io import savemat
                savemat(filename, data)
            except ImportError:
                raise ImportError("scipy is required for .mat export")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return True
    
    except Exception as e:
        warnings.warn(f"Export failed: {str(e)}")
        return False


def import_data(filename: str, format: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Import data from various formats.
    
    Parameters:
        filename: Input filename
        format: Import format (auto-detected if None)
        
    Returns:
        Dictionary of imported arrays
    """
    if format is None:
        # Auto-detect format from extension
        if filename.endswith('.npz'):
            format = 'npz'
        elif filename.endswith('.csv'):
            format = 'csv'
        elif filename.endswith('.mat'):
            format = 'mat'
        else:
            raise ValueError("Cannot auto-detect format. Please specify format parameter.")
    
    try:
        if format == 'npz':
            with np.load(filename) as data:
                return dict(data)
        elif format == 'csv':
            import pandas as pd
            df = pd.read_csv(filename)
            return {col: df[col].values for col in df.columns}
        elif format == 'mat':
            try:
                from scipy.io import loadmat
                data = loadmat(filename)
                # Remove scipy metadata
                return {k: v for k, v in data.items() if not k.startswith('__')}
            except ImportError:
                raise ImportError("scipy is required for .mat import")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    except Exception as e:
        raise IOError(f"Import failed: {str(e)}")


def create_report(data: np.ndarray, H: Optional[float] = None, 
                 output_file: Optional[str] = None) -> str:
    """
    Create a comprehensive analysis report for a time series.
    
    Parameters:
        data: Time series data
        H: Known Hurst exponent (optional)
        output_file: Output filename for report (optional)
        
    Returns:
        Report as string
    """
    from .analysis import estimate_hurst_multiple_methods, validate_fgn_properties
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("FRACTALSIG ANALYSIS REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    
    # Basic statistics
    report_lines.append("BASIC STATISTICS:")
    report_lines.append(f"  Length: {len(data)}")
    report_lines.append(f"  Mean: {np.mean(data):.6f}")
    report_lines.append(f"  Standard Deviation: {np.std(data):.6f}")
    report_lines.append(f"  Variance: {np.var(data):.6f}")
    report_lines.append(f"  Minimum: {np.min(data):.6f}")
    report_lines.append(f"  Maximum: {np.max(data):.6f}")
    report_lines.append(f"  Range: {np.ptp(data):.6f}")
    report_lines.append("")
    
    # Hurst exponent estimation
    report_lines.append("HURST EXPONENT ESTIMATION:")
    try:
        hurst_results = estimate_hurst_multiple_methods(data)
        
        for method, result in hurst_results.items():
            if method == 'summary':
                continue
            if 'H' in result:
                report_lines.append(f"  {result['method']}: H = {result['H']:.4f}")
            else:
                report_lines.append(f"  {result.get('method', method)}: Failed ({result.get('error', 'Unknown error')})")
        
        if 'summary' in hurst_results:
            summary = hurst_results['summary']
            report_lines.append(f"  Summary (n={summary['n_methods']}):")
            report_lines.append(f"    Mean H: {summary['mean_H']:.4f}")
            report_lines.append(f"    Std H:  {summary['std_H']:.4f}")
            report_lines.append(f"    Range:  [{summary['min_H']:.4f}, {summary['max_H']:.4f}]")
    
    except Exception as e:
        report_lines.append(f"  Error in Hurst estimation: {str(e)}")
    
    report_lines.append("")
    
    # Validation (if H is provided)
    if H is not None:
        report_lines.append("VALIDATION RESULTS:")
        try:
            validation = validate_fgn_properties(data, H)
            
            if 'normality' in validation:
                norm_result = validation['normality']
                report_lines.append(f"  Normality ({norm_result['test']}): {'PASS' if norm_result['is_normal'] else 'FAIL'}")
                if 'p_value' in norm_result:
                    report_lines.append(f"    p-value: {norm_result['p_value']:.4f}")
            
            if 'stationarity' in validation:
                stat_result = validation['stationarity']
                report_lines.append(f"  Stationarity: {'PASS' if stat_result['is_stationary'] else 'FAIL'}")
            
            if 'hurst_validation' in validation:
                hurst_val = validation['hurst_validation']
                if 'error' not in hurst_val:
                    report_lines.append(f"  Hurst Accuracy: {'PASS' if hurst_val['is_accurate'] else 'FAIL'}")
                    report_lines.append(f"    Expected: {hurst_val['expected_H']:.4f}")
                    report_lines.append(f"    Estimated: {hurst_val['estimated_H']:.4f}")
                    report_lines.append(f"    Error: {hurst_val['error']:.4f}")
        
        except Exception as e:
            report_lines.append(f"  Error in validation: {str(e)}")
        
        report_lines.append("")
    
    # Spectral properties
    report_lines.append("SPECTRAL PROPERTIES:")
    try:
        from .core import fft
        freqs, magnitudes = fft(data)
        
        # Find dominant frequency
        positive_freqs = freqs[:len(freqs)//2]
        positive_mags = magnitudes[:len(magnitudes)//2]
        
        if len(positive_mags) > 1:
            peak_idx = np.argmax(positive_mags[1:]) + 1  # Skip DC
            dominant_freq = positive_freqs[peak_idx]
            report_lines.append(f"  Dominant Frequency: {dominant_freq:.6f}")
            report_lines.append(f"  Peak Magnitude: {positive_mags[peak_idx]:.4f}")
            report_lines.append(f"  DC Component: {magnitudes[0]:.4f}")
    
    except Exception as e:
        report_lines.append(f"  Error in spectral analysis: {str(e)}")
    
    report_lines.append("")
    
    # Footer
    report_lines.append("="*60)
    report_lines.append("Report generated by FractalSig library")
    report_lines.append("="*60)
    
    # Combine into single string
    report = "\n".join(report_lines)
    
    # Save to file if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report)
        except Exception as e:
            warnings.warn(f"Could not save report to file: {str(e)}")
    
    return report


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and reproducibility.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'numpy_version': np.__version__,
    }
    
    # Try to get additional package versions
    try:
        import scipy
        info['scipy_version'] = scipy.__version__
    except ImportError:
        info['scipy_version'] = 'Not installed'
    
    try:
        import pywt
        info['pywavelets_version'] = pywt.__version__
    except ImportError:
        info['pywavelets_version'] = 'Not installed'
    
    try:
        import matplotlib
        info['matplotlib_version'] = matplotlib.__version__
    except ImportError:
        info['matplotlib_version'] = 'Not installed'
    
    return info 