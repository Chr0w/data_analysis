#!/bin/bash
# Script to set up virtual environment and run the plotting script

set -e  # Exit on error

# Get the script's directory (works from any location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use venv's pip and python directly
VENV_PIP="$SCRIPT_DIR/venv/bin/pip"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

# Check if venv exists and is valid
VENV_VALID=false
if [ -d "venv" ] && [ -f "$VENV_PYTHON" ] && [ -f "$VENV_PIP" ]; then
    # Check if pip's shebang points to the correct path
    # Read the first line of pip to check the shebang (remove #! prefix)
    PIP_SHEBANG=$(head -n 1 "$VENV_PIP" 2>/dev/null | sed 's/^#!//' | tr -d ' ' || echo "")
    if [ -n "$PIP_SHEBANG" ]; then
        # Check if the shebang path contains the current script directory
        # This works for both absolute paths and relative paths
        if echo "$PIP_SHEBANG" | grep -q "$SCRIPT_DIR" || [ "$PIP_SHEBANG" = "$VENV_PYTHON" ]; then
            # Also test if pip actually works
            if "$VENV_PIP" --version >/dev/null 2>&1; then
                VENV_VALID=true
            else
                echo "Virtual environment's pip is not working. Recreating..."
                rm -rf venv
            fi
        else
            echo "Virtual environment was created with a different path (found: $PIP_SHEBANG). Recreating..."
            rm -rf venv
        fi
    else
        # If we can't read the shebang, test if pip works
        if "$VENV_PIP" --version >/dev/null 2>&1; then
            VENV_VALID=true
        else
            echo "Virtual environment's pip is not working. Recreating..."
            rm -rf venv
        fi
    fi
fi

# Create virtual environment if it doesn't exist or is invalid
if [ "$VENV_VALID" = false ]; then
    echo "Creating virtual environment..."
    if ! python3 -m venv venv; then
        echo "Error: Failed to create virtual environment. Make sure python3-venv is installed."
        exit 1
    fi
    # Update paths after recreation
    VENV_PIP="$SCRIPT_DIR/venv/bin/pip"
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
fi

# Install requirements
echo "Installing requirements..."
$VENV_PIP install --upgrade pip
$VENV_PIP install -r requirements.txt

# Run the plotting script
echo "Running plotting script..."
$VENV_PYTHON plot_data.py
