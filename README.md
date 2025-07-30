# FractalSig

A Python library for generating and analyzing fractional Gaussian noise (fGn) and fractional Brownian motion (fBm) using advanced mathematical methods.

## Features

- **fgn**: Generate fractional Gaussian noise using the Davies-Harte method
- **fbm**: Compute fractional Brownian motion from fGn
- **fft**: Fast Fourier Transform analysis
- **fwt**: Fast Wavelet Transform using PyWavelets
- **Analysis**: R/S analysis, DFA, wavelet-based Hurst estimation, and more
- **Visualization**: Professional plotting functions for all analyses
- **Utilities**: Performance benchmarking, validation, and reporting tools

## Installation

### Option 1: Virtual Environment (Recommended)
```bash
# Create and activate virtual environment
python -m venv fractalsig-env
source fractalsig-env/bin/activate  # On Windows: fractalsig-env\Scripts\activate

# Install the package
pip install -e .
```

### Option 2: User Installation
```bash
pip install --user -e .
```

### Option 3: Development Setup
Use the provided setup scripts:
```bash
# Linux/macOS
./setup_dev.sh

# Windows
setup_dev.bat
```

## Quick Start

```python
from fractalsig import fgn, fbm, fft, fwt

# Generate fractional Gaussian noise
H = 0.7  # Hurst exponent
L = 1024  # Length (power of 2 recommended for optimal FFT performance)
fgn_data = fgn(H, L)

# Convert to fractional Brownian motion
fbm_data = fbm(fgn_data)

# Analyze with FFT
freqs, magnitudes = fft(fgn_data)

# Wavelet analysis
coeffs = fwt(fgn_data, wavelet='db4')
```

## Core Functions

### fgn(H, L)
Generate fractional Gaussian noise using the Davies-Harte method.

- **H**: Hurst exponent (0 < H < 1)
- **L**: Length of time series (power of 2 recommended)
- **Returns**: Array of fractional Gaussian noise

### fbm(data)
Compute fractional Brownian motion from fractional Gaussian noise.

- **data**: Input fGn data
- **Returns**: fBm array (length = len(data) + 1, starts at 0)

### fft(data)
Fast Fourier Transform analysis.

- **data**: Input signal
- **Returns**: (frequencies, magnitudes) tuple

### fwt(data, wavelet='db2', level=None)
Fast Wavelet Transform using PyWavelets.

- **data**: Input signal
- **wavelet**: Wavelet type (default: 'db2')
- **level**: Decomposition level (auto-determined if None)
- **Returns**: List of wavelet coefficients

## Analysis Functions

The library includes comprehensive analysis capabilities:

- **rs_analysis**: Rescaled Range analysis for Hurst estimation
- **dfa_analysis**: Detrended Fluctuation Analysis
- **wavelet_hurst_estimation**: Wavelet-based Hurst estimation
- **estimate_hurst_multiple_methods**: Compare multiple estimation methods
- **autocorrelation_function**: Compute autocorrelation
- **power_spectral_density**: Power spectral density estimation

## Visualization Functions

Professional plotting capabilities:

- **plot_fgn**: Plot fractional Gaussian noise
- **plot_fbm**: Plot fractional Brownian motion
- **plot_fft_spectrum**: FFT spectrum visualization
- **plot_wavelet_coefficients**: Wavelet coefficient visualization
- **plot_hurst_comparison**: Compare Hurst estimation methods
- **plot_summary**: Comprehensive analysis summary

## Utility Functions

Performance and validation tools:

- **benchmark_fgn_methods**: Compare fGn generation methods
- **validate_algorithm_correctness**: Verify implementation correctness
- **generate_test_dataset**: Create synthetic test data
- **create_report**: Generate comprehensive analysis reports

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Demo

Run the demonstration script:
```python
python demo.py
```

This generates example plots showing fGn, fBm, FFT, and wavelet analysis.

## Documentation

For detailed mathematical theory and implementation details, see the LaTeX documentation in the `docs/` folder:

```bash
cd docs/
make pdf    
```

## Mathematical Background

This library implements:

- **Davies-Harte Method**: Efficient fGn generation using circulant embedding
- **Fractional Brownian Motion**: Self-similar Gaussian process
- **R/S Analysis**: Rescaled Range statistical method
- **Wavelet Analysis**: Multi-resolution signal decomposition
- **Spectral Analysis**: Frequency domain characterization

The Hurst exponent H controls the roughness and long-range dependence:
- H = 0.5: Standard Brownian motion (no correlation)
- H > 0.5: Persistent behavior (positive correlation)
- H < 0.5: Anti-persistent behavior (negative correlation)

## Requirements

- Python 3.7+
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- PyWavelets ≥ 1.2.0

## License

MIT License