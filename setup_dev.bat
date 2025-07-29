@echo off
REM FractalSig Development Setup Script for Windows
REM This script creates a virtual environment and installs the package in development mode

echo 🔧 Setting up FractalSig development environment...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not found. Please install Python 3.7 or later.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
set VENV_DIR=fractalsig-env
if not exist "%VENV_DIR%" (
    echo 📦 Creating virtual environment...
    python -m venv %VENV_DIR%
) else (
    echo 📦 Virtual environment already exists.
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install package in development mode
echo 📥 Installing fractalsig in development mode...
pip install -e .

REM Install development dependencies
echo 🧪 Installing test dependencies...
pip install pytest matplotlib

echo.
echo ✅ Setup complete!
echo.
echo To activate the environment in the future, run:
echo   %VENV_DIR%\Scripts\activate.bat
echo.
echo To run tests:
echo   python -m pytest tests/ -v
echo.
echo To run the demo:
echo   python demo.py
echo.
echo To deactivate when done:
echo   deactivate
echo.
pause 