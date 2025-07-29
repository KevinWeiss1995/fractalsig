"""
FractalSig: Python library for generating and analyzing fractional Gaussian noise and related transforms.
"""

# Core functions
from .core import fgn, fbm, fft, fwt

# Analysis functions
from .analysis import (
    rs_analysis, dfa_analysis, wavelet_hurst_estimation,
    estimate_hurst_multiple_methods, autocorrelation_function,
    power_spectral_density, validate_fgn_properties,
    generate_confidence_intervals, compare_time_series
)

# Plotting functions
from .plotting import (
    plot_fgn, plot_fbm, plot_fft_spectrum, plot_wavelet_coefficients,
    plot_hurst_comparison, plot_rs_analysis, plot_autocorrelation,
    plot_summary
)

# Utility functions
from .utils import (
    benchmark_fgn_methods, validate_algorithm_correctness,
    generate_test_dataset, memory_usage_profile,
    export_data, import_data, create_report, get_system_info
)

__version__ = "0.1.0"

# Core functions
__all__ = ["fgn", "fbm", "fft", "fwt"]

# Analysis functions
__all__ += [
    "rs_analysis", "dfa_analysis", "wavelet_hurst_estimation",
    "estimate_hurst_multiple_methods", "autocorrelation_function",
    "power_spectral_density", "validate_fgn_properties",
    "generate_confidence_intervals", "compare_time_series"
]

# Plotting functions  
__all__ += [
    "plot_fgn", "plot_fbm", "plot_fft_spectrum", "plot_wavelet_coefficients",
    "plot_hurst_comparison", "plot_rs_analysis", "plot_autocorrelation",
    "plot_summary"
]

# Utility functions
__all__ += [
    "benchmark_fgn_methods", "validate_algorithm_correctness",
    "generate_test_dataset", "memory_usage_profile",
    "export_data", "import_data", "create_report", "get_system_info"
] 