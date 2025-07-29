# FractalSig Documentation

This directory contains comprehensive technical documentation for the FractalSig library.

## Contents

- `fractalsig_theory_implementation.tex` - Complete LaTeX document covering mathematical theory and implementation details
- `compile.sh` - Compilation script for generating the PDF
- `Makefile` - Make targets for easy compilation and cleanup

## Compiling the Documentation

### Prerequisites

You need a LaTeX distribution installed:

**macOS:**
```bash
brew install --cask mactex
```

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
```

**Windows:**
Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

### Compilation Options

#### Option 1: Using the script
```bash
cd docs/
./compile.sh
```

#### Option 2: Using Make
```bash
cd docs/
make          # Compile PDF
make open     # Compile and open PDF
make clean    # Remove auxiliary files
```

#### Option 3: Manual compilation
```bash
cd docs/
pdflatex fractalsig_theory_implementation.tex
pdflatex fractalsig_theory_implementation.tex  # Second pass for cross-references
```

## Document Contents

The comprehensive document covers:

### Mathematical Theory
- Fractional Gaussian noise (fGn) definition and properties
- Fractional Brownian motion (fBm) characteristics
- Hurst exponent interpretation
- Spectral density and long-range dependence
- Self-similarity and scaling properties

### Implementation Details
- Davies-Harte algorithm with circulant embedding
- Cholesky decomposition fallback method
- FFT analysis implementation
- Wavelet transform using PyWavelets
- Numerical stability considerations

### Validation and Testing
- R/S analysis for Hurst exponent estimation
- Comprehensive test suite architecture
- Performance benchmarks and complexity analysis
- Error handling and robustness mechanisms

### Code Analysis  
- Line-by-line implementation explanations
- Algorithm complexity analysis
- Memory usage optimization
- Cross-platform compatibility

This document provides everything needed to fully understand both the mathematical foundations and practical implementation choices in the FractalSig library. 