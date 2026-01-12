#!/bin/bash
# Script to set up virtual environment and run the plotting script

set -e  # Exit on error

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    if ! python3 -m venv venv; then
        echo "Error: Failed to create virtual environment. Make sure python3-venv is installed."
        exit 1
    fi
fi

# Use venv's pip and python directly
VENV_PIP="venv/bin/pip"
VENV_PYTHON="venv/bin/python3"

# Check if venv was created successfully
if [ ! -f "$VENV_PIP" ] || [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment is incomplete. Removing and recreating..."
    rm -rf venv
    python3 -m venv venv
fi

# Install requirements
echo "Installing requirements..."
$VENV_PIP install --upgrade pip
$VENV_PIP install -r requirements.txt

# Run the plotting script
echo "Running plotting script..."
$VENV_PYTHON plot_data.py
