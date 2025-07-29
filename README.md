# FractalSig

Python library for generating and analyzing fractional Gaussian noise and related transforms.

## Features

- **fgn**: Generate fractional Gaussian noise using Davies-Harte method
- **fbn**: Compute fractional Brownian motion from fGn  
- **fft**: Fast Fourier Transform with frequency analysis
- **fwt**: Fast Wavelet Transform using PyWavelets

## Installation

### Quick Setup (Recommended)
Use the provided setup scripts to avoid permission issues:

**Linux/macOS:**
```bash
./setup_dev.sh
```

**Windows:**
```cmd
setup_dev.bat
```

### Manual Installation

#### Option 1: Virtual Environment
```bash
# Create and activate virtual environment
python -m venv fractalsig-env
source fractalsig-env/bin/activate  # On Windows: fractalsig-env\Scripts\activate

# Install in development mode
pip install -e .
```

#### Option 2: User Installation (No sudo required)
```bash
pip install -e . --user
```

#### Option 3: From PyPI (when published)
```bash
pip install fractalsig
```

## Requirements

- numpy >= 1.20.0
- scipy >= 1.7.0  
- PyWavelets >= 1.2.0

## Quick Start

```python
import numpy as np
from fractalsig import fgn, fbn, fft, fwt

# Generate fractional Gaussian noise
H = 0.7  # Hurst exponent (0 < H < 1)
L = 1024  # Length
fgn_data = fgn(H, L)

# Convert to fractional Brownian motion
fbm_data = fbn(fgn_data)

# Analyze with FFT
freqs, magnitudes = fft(fgn_data)

# Analyze with wavelets
coeffs = fwt(fgn_data, wavelet='db4')
```

## Function Reference

### fgn(H, L)
Generate fractional Gaussian noise.

**Parameters:**
- `H` (float): Hurst exponent (0 < H < 1)
- `L` (int): Length of time series

**Returns:** np.ndarray of shape (L,)

### fbn(data)
Compute fractional Brownian motion from fGn.

**Parameters:**
- `data` (np.ndarray): 1D input data

**Returns:** np.ndarray of shape (len(data) + 1,)

### fft(data)
Fast Fourier Transform analysis.

**Parameters:**
- `data` (np.ndarray): 1D input signal

**Returns:** Tuple (freqs, magnitudes)

### fwt(data, wavelet='db2', level=None)
Fast Wavelet Transform.

**Parameters:**
- `data` (np.ndarray): 1D input signal
- `wavelet` (str): Wavelet type (default: 'db2')
- `level` (int): Decomposition level (auto if None)

**Returns:** List of coefficient arrays

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Demo

Run the demonstration script:

```bash
python demo.py
```

This generates example plots showing all library functions in action.

## Documentation

Comprehensive technical documentation is available in the `docs/` directory:

```bash
cd docs/
make          # Compile LaTeX documentation to PDF
make open     # Compile and open the PDF
```

The documentation covers:
- **Mathematical theory**: fGn, fBm, Hurst exponents, spectral properties
- **Implementation details**: Davies-Harte algorithm, numerical methods
- **Code analysis**: Line-by-line explanations, complexity analysis
- **Validation methods**: R/S analysis, testing architecture

Prerequisites: LaTeX distribution (MacTeX, TeX Live, or MiKTeX)