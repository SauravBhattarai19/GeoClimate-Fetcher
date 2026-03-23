#!/bin/bash
# ========================================================
#  GeoClimate Intelligence Platform - Linux Installer
# ========================================================
#
#  Run this script:
#    chmod +x install-linux.sh
#    ./install-linux.sh
#
#  Or right-click the file > "Run as Program" / "Run in Terminal"
#
# ========================================================

set -euo pipefail

# Navigate to this script's directory
cd "$(dirname "$0")"

echo ""
echo "========================================================"
echo "   GeoClimate Intelligence Platform - Linux Installer"
echo "========================================================"
echo ""
echo " This will set up everything you need to run GeoClimate"
echo " Fetcher on your system. You may be asked for your"
echo " password to install system packages."
echo ""
echo "========================================================"
echo ""

# ========================================
#  Helper: Detect distro and package manager
# ========================================
detect_package_manager() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO_ID="${ID:-unknown}"
        DISTRO_ID_LIKE="${ID_LIKE:-}"
    else
        DISTRO_ID="unknown"
        DISTRO_ID_LIKE=""
    fi

    if command -v apt-get &>/dev/null; then
        PKG_MANAGER="apt"
    elif command -v dnf &>/dev/null; then
        PKG_MANAGER="dnf"
    elif command -v pacman &>/dev/null; then
        PKG_MANAGER="pacman"
    elif command -v zypper &>/dev/null; then
        PKG_MANAGER="zypper"
    else
        PKG_MANAGER="unknown"
    fi
}

install_packages_apt() {
    echo "   Detected Debian/Ubuntu-based system."
    sudo apt-get update -qq

    # Install git if needed
    if ! command -v git &>/dev/null; then
        echo "   Installing Git..."
        sudo apt-get install -y git
    fi

    # Check if python3.11 is available
    if ! apt-cache show python3.11 &>/dev/null 2>&1; then
        echo "   Python 3.11 not in default repos. Adding deadsnakes PPA..."
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update -qq
    fi

    echo "   Installing Python 3.11 and venv..."
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
}

install_packages_dnf() {
    echo "   Detected Fedora/RHEL-based system."

    if ! command -v git &>/dev/null; then
        echo "   Installing Git..."
        sudo dnf install -y git
    fi

    echo "   Installing Python 3.11..."
    sudo dnf install -y python3.11 python3.11-devel
}

install_packages_pacman() {
    echo "   Detected Arch-based system."

    if ! command -v git &>/dev/null; then
        echo "   Installing Git..."
        sudo pacman -Sy --noconfirm git
    fi

    echo "   Installing Python..."
    sudo pacman -Sy --noconfirm python
}

install_packages_zypper() {
    echo "   Detected openSUSE-based system."

    if ! command -v git &>/dev/null; then
        echo "   Installing Git..."
        sudo zypper install -y git
    fi

    echo "   Installing Python 3.11..."
    sudo zypper install -y python311 python311-devel
}

detect_package_manager

# ========================================
#  STEP 1: Check / Install Git
# ========================================
echo "[Step 1/6] Checking for Git..."
if command -v git &>/dev/null; then
    echo "   Git is already installed."
else
    echo "   Git is not installed. Installing..."
    case "$PKG_MANAGER" in
        apt)     sudo apt-get update -qq && sudo apt-get install -y git ;;
        dnf)     sudo dnf install -y git ;;
        pacman)  sudo pacman -Sy --noconfirm git ;;
        zypper)  sudo zypper install -y git ;;
        *)
            echo "   ERROR: Could not detect package manager."
            echo "   Please install git manually, then re-run this script."
            exit 1
            ;;
    esac
    echo "   Git installed."
fi

# ========================================
#  STEP 2: Clone repository if needed
# ========================================
echo ""
echo "[Step 2/6] Checking for project files..."
if [ -f "app.py" ]; then
    echo "   Project files found. Skipping clone."
else
    echo "   Project files not found. Cloning repository..."
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
    echo "   Python 3.11+ not found. Installing..."

    # Check for sudo access
    if ! command -v sudo &>/dev/null || ! sudo -v 2>/dev/null; then
        echo ""
        echo "   ERROR: sudo access is required to install Python."
        echo "   Please ask your system administrator to install these packages:"
        echo ""
        case "$PKG_MANAGER" in
            apt)     echo "     sudo apt install python3.11 python3.11-venv python3.11-dev" ;;
            dnf)     echo "     sudo dnf install python3.11 python3.11-devel" ;;
            pacman)  echo "     sudo pacman -S python" ;;
            zypper)  echo "     sudo zypper install python311 python311-devel" ;;
            *)       echo "     Install Python 3.11+ from https://www.python.org/downloads/" ;;
        esac
        echo ""
        echo "   Then re-run this script."
        exit 1
    fi

    case "$PKG_MANAGER" in
        apt)     install_packages_apt ;;
        dnf)     install_packages_dnf ;;
        pacman)  install_packages_pacman ;;
        zypper)  install_packages_zypper ;;
        *)
            echo "   ERROR: Unsupported package manager."
            echo "   Please install Python 3.11+ from https://www.python.org/downloads/"
            echo "   Then re-run this script."
            exit 1
            ;;
    esac

    # Re-detect Python after install
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
        echo "   ERROR: Python 3.11+ still not found after installation."
        echo "   Please install manually from https://www.python.org/downloads/"
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
echo "   For future launches, run: ./run-linux.sh"
echo "========================================================"
echo ""

streamlit run app.py
