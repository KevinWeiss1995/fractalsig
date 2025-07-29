#!/bin/bash

# FractalSig Development Setup Script
# This script creates a virtual environment and installs the package in development mode

set -e  # Exit on any error

echo "🔧 Setting up FractalSig development environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python 3.7 or later."
    exit 1
fi

# Create virtual environment if it doesn't exist
VENV_DIR="fractalsig-env"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv $VENV_DIR
else
    echo "📦 Virtual environment already exists."
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "📥 Installing fractalsig in development mode..."
pip install -e .

# Install development dependencies
echo "🧪 Installing test dependencies..."
pip install pytest matplotlib

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run tests:"
echo "  python -m pytest tests/ -v"
echo ""
echo "To run the demo:"
echo "  python demo.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate" 