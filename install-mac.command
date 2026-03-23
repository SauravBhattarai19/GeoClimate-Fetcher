#!/bin/bash
# ========================================================
#  GeoClimate Intelligence Platform - macOS Installer
# ========================================================
#
#  Double-click this file in Finder to run it.
#  If macOS blocks it: Right-click > Open > Open
#
# ========================================================

set -euo pipefail

# Navigate to this script's directory
cd "$(dirname "$0")"

echo ""
echo "========================================================"
echo "   GeoClimate Intelligence Platform - macOS Installer"
echo "========================================================"
echo ""
echo " This will set up everything you need to run GeoClimate"
echo " Fetcher on your Mac. Just follow any prompts that appear."
echo ""
echo "========================================================"
echo ""

# ========================================
#  STEP 1: Check / Install Git
# ========================================
echo "[Step 1/6] Checking for Git..."
if command -v git &>/dev/null; then
    echo "   Git is already installed."
else
    echo "   Git is not installed. Installing Xcode Command Line Tools..."
    echo "   A dialog box will appear — click 'Install' and wait for it to finish."
    echo ""
    xcode-select --install 2>/dev/null || true
    echo ""
    echo "   After the installation dialog completes, press ENTER to continue..."
    read -r
    if ! command -v git &>/dev/null; then
        echo "   ERROR: Git is still not available."
        echo "   Please complete the Xcode Command Line Tools installation,"
        echo "   then double-click this file again."
        exit 1
    fi
fi

# ========================================
#  STEP 2: Clone repository if needed
# ========================================
echo ""
echo "[Step 2/6] Checking for project files..."
if [ -f "app.py" ]; then
    echo "   Project files found. Skipping clone."
elif [ -f "GeoClimate-Fetcher/app.py" ]; then
    echo "   Found existing GeoClimate-Fetcher folder. Using it."
    cd GeoClimate-Fetcher
else
    echo "   Project files not found. Cloning repository..."
    if [ -d "GeoClimate-Fetcher" ]; then
        echo "   Note: GeoClimate-Fetcher folder exists but appears incomplete."
        echo "   Removing it and cloning fresh..."
        rm -rf GeoClimate-Fetcher
    fi
    git clone https://github.com/SauravBhattarai19/GeoClimate-Fetcher.git
    cd GeoClimate-Fetcher
    echo "   Repository cloned successfully."
fi

# ========================================
#  STEP 3: Check / Install Python 3.11+
# ========================================
echo ""
echo "[Step 3/6] Checking for Python 3.11+..."

PYTHON_CMD=""

# Find a suitable Python >= 3.11
for cmd in python3.11 python3.12 python3.13 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 | awk '{print $2}')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "   Python 3.11+ not found."

    if command -v brew &>/dev/null; then
        echo "   Homebrew detected. Installing Python 3.11..."
        brew install python@3.11
        PYTHON_CMD="python3.11"
    else
        echo ""
        echo "   Please install Python 3.11+ using one of these methods:"
        echo ""
        echo "   Option A (recommended): Install Homebrew first, then Python:"
        echo "     /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "     brew install python@3.11"
        echo ""
        echo "   Option B: Download from https://www.python.org/downloads/"
        echo ""
        echo "   After installing Python, double-click this file again."
        exit 1
    fi
fi

echo "   Found $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"

# ========================================
#  STEP 4: Create virtual environment
# ========================================
echo ""
echo "[Step 4/6] Setting up virtual environment..."
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    echo "   Virtual environment already exists."
else
    echo "   Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    echo "   Virtual environment created."
fi

# Activate venv
source .venv/bin/activate

# ========================================
#  STEP 5: Install dependencies
# ========================================
echo ""
echo "[Step 5/6] Installing dependencies (this may take a few minutes)..."
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo ""
    echo "   ERROR: Dependency installation failed."
    echo "   Try running manually:"
    echo "     source .venv/bin/activate"
    echo "     pip install -r requirements.txt"
    exit 1
fi
echo "   Dependencies installed successfully."

# ========================================
#  STEP 6: Earth Engine Authentication
# ========================================
echo ""
echo "[Step 6/6] Google Earth Engine Authentication"
echo ""
echo "========================================================"
echo " IMPORTANT: A browser window will open."
echo " Log in with the Google account you used to sign up"
echo " for Google Earth Engine."
echo ""
echo " Don't have an account? Sign up FREE at:"
echo " https://earthengine.google.com/signup/"
echo ""
echo " Press Ctrl+C to skip and authenticate later."
echo "========================================================"
echo ""

CRED_FILE="$HOME/.config/earthengine/credentials"

if [ -f "$CRED_FILE" ]; then
    echo "   Earth Engine credentials found. Skipping authentication."
    echo "   (Delete $CRED_FILE to re-authenticate)"
else
    earthengine authenticate || {
        echo ""
        echo "   Warning: Earth Engine authentication was not completed."
        echo "   You can authenticate later by running: earthengine authenticate"
        echo "   Or upload credentials through the app interface."
    }
fi

# ========================================
#  LAUNCH THE APPLICATION
# ========================================
echo ""
echo "========================================================"
echo "   Setup complete! Launching GeoClimate Fetcher..."
echo "   The app will open in your web browser shortly."
echo ""
echo "   For future launches, double-click: run-mac.command"
echo "========================================================"
echo ""

streamlit run app.py
