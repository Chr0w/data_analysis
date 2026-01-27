#!/bin/bash
# Script to set up and activate virtual environment

# Get the script's directory (works from any location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    if ! python3 -m venv venv; then
        echo "Error: Failed to create virtual environment. Make sure python3-venv is installed."
        exit 1
    fi
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Virtual environment is now active!"
echo "You can now install packages with: pip install <package>"
echo "To deactivate, run: deactivate"
echo ""
echo "Note: If you ran this script with ./setup_and_run.sh, the venv activation"
echo "      only applies to this script's execution. To activate in your current shell,"
echo "      run: source setup_and_run.sh"
echo "      or: source venv/bin/activate"
