#!/bin/bash

# LaTeX compilation script for FractalSig documentation
# Compiles with proper bibliography and cross-references

set -e

MAIN_FILE="fractalsig_theory_implementation"
TEX_FILE="${MAIN_FILE}.tex"
PDF_FILE="${MAIN_FILE}.pdf"

echo "🔧 Compiling FractalSig documentation..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ pdflatex not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)"
    echo "   On macOS: brew install --cask mactex"
    echo "   On Ubuntu: sudo apt-get install texlive-full"
    exit 1
fi

# First pass - generate aux files
echo "📄 First LaTeX pass..."
pdflatex -interaction=nonstopmode "$TEX_FILE" > /dev/null

# Second pass - resolve cross-references
echo "📄 Second LaTeX pass..."
pdflatex -interaction=nonstopmode "$TEX_FILE" > /dev/null

# Clean up auxiliary files
echo "🧹 Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk

if [ -f "$PDF_FILE" ]; then
    echo "✅ Documentation compiled successfully: $PDF_FILE"
    
    # Open PDF if on macOS
    if command -v open &> /dev/null; then
        echo "📖 Opening PDF..."
        open "$PDF_FILE"
    fi
else
    echo "❌ Compilation failed. Check for LaTeX errors."
    exit 1
fi 