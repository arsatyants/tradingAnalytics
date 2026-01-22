#!/bin/bash
# Trading Analytics Environment Setup Script
# ===========================================
# Automatic setup: Creates .venv and installs all dependencies + GPU support

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  Trading Analytics - Automated Environment Setup"
echo "======================================================================"
echo ""

# Check for Orange Pi
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model 2>/dev/null)
    if [[ $MODEL == *"OrangePi"* ]]; then
        echo -e "${YELLOW}⚠ IMPORTANT: GPU Setup Required${NC}"
        echo ""
        echo "Before continuing, please enable the GPU in orangepi-config:"
        echo -e "  1. Run: ${BLUE}sudo orangepi-config${NC}"
        echo "  2. Navigate to: System → Hardware"
        echo "  3. Enable: gpu_mali (panfrost driver)"
        echo "  4. Save and reboot if needed"
        echo ""
        read -p "Press Enter to continue after enabling GPU (or skip if already done)..."
        echo ""
    fi
fi

# Find Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}✗ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}Python version:${NC} $PYTHON_VERSION"
echo ""

# Virtual environment directory
VENV_DIR=".venv"

# Remove existing venv if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create virtual environment
echo -e "${BLUE}[1/6]${NC} Creating virtual environment..."
$PYTHON_CMD -m venv "$VENV_DIR"
echo -e "${GREEN}✓${NC} Created: $VENV_DIR/"

# Activate virtual environment
echo ""
echo -e "${BLUE}[2/6]${NC} Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓${NC} Activated"

# Upgrade pip
echo ""
echo -e "${BLUE}[3/6]${NC} Upgrading pip..."
pip install --upgrade pip -q
echo -e "${GREEN}✓${NC} pip upgraded"

# Install all dependencies
echo ""
echo -e "${BLUE}[4/6]${NC} Installing all dependencies from requirements.txt..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo ""
echo -e "${GREEN}✓${NC} All dependencies installed

# GPU Setup (OpenCL & Vulkan)
echo ""
echo -e "${BLUE}[5/6]${NC} Setting up GPU support (OpenCL & Vulkan)..."
echo ""

# Install OpenCL support
echo "Installing OpenCL runtime..."
sudo apt update -qq
sudo apt install -y ocl-icd-libopencl1 mesa-opencl-icd clinfo >/dev/null 2>&1
echo -e "${GREEN}✓${NC} OpenCL installed"

# Install Vulkan support
echo "Installing Vulkan runtime..."
sudo apt install -y vulkan-tools mesa-vulkan-drivers libvulkan1 >/dev/null 2>&1
echo -e "${GREEN}✓${NC} Vulkan installed"

# Install Mesa utilities
echo "Installing Mesa GPU utilities..."
sudo apt install -y mesa-utils libgl1-mesa-dri >/dev/null 2>&1
echo -e "${GREEN}✓${NC} Mesa utilities installed"

# Load panfrost GPU driver
echo "Loading panfrost GPU driver..."
sudo modprobe panfrost 2>/dev/null || echo -e "${YELLOW}⚠${NC} Panfrost module not loaded (enable GPU in orangepi-config)"
echo "panfrost" | sudo tee /etc/modules-load.d/panfrost.conf >/dev/null
echo -e "${GREEN}✓${NC} Panfrost configured to load on boot"

# Set GPU permissions
echo "Configuring GPU permissions..."
sudo usermod -a -G render $USER 2>/dev/null || true
sudo usermod -a -G video $USER 2>/dev/null || true
echo -e "${GREEN}✓${NC} User added to render and video groups"

# Configure environment
echo ""
echo -e "${BLUE}[6/6]${NC} Configuring GPU environment..."

# Add to venv activation
echo 'export RUSTICL_ENABLE=panfrost' >> "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓${NC} RUSTICL_ENABLE=panfrost added to venv activation"

# Add to user's .bashrc for system-wide availability
if ! grep -q "RUSTICL_ENABLE=panfrost" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# GPU acceleration for Trading Analytics" >> ~/.bashrc
    echo "export RUSTICL_ENABLE=panfrost" >> ~/.bashrc
    echo -e "${GREEN}✓${NC} Added RUSTICL_ENABLE to ~/.bashrc"
else
    echo -e "${GREEN}✓${NC} RUSTICL_ENABLE already in ~/.bashrc"
fi

# Make helper scripts executable
chmod +x run_gpu.sh run_web.sh 2>/dev/null || true
echo -e "${GREEN}✓${NC} Helper scripts made executable"

# Test GPU
echo ""
echo "Testing GPU availability..."
if [ -e /dev/dri/card0 ]; then
    echo -e "${GREEN}✓${NC} GPU device found: /dev/dri/card0"
    RUSTICL_ENABLE=panfrost clinfo 2>/dev/null | grep -i "Mali" && echo -e "${GREEN}✓${NC} Mali GPU detected by OpenCL" || echo -e "${YELLOW}⚠${NC} GPU not detected by OpenCL (reboot may be needed)"
else
    echo -e "${YELLOW}⚠${NC} GPU device not found - enable in orangepi-config and reboot"
fi"

# Summary
echo ""
echo "======================================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "======================================================================"
echo ""
echo "Virtual environment: $VENV_DIR/"
echo ""
echo "GPU Status:"
if [ -e /dev/dri/card0 ]; then
    echo -e "  ${GREEN}✓${NC} GPU device detected"
    echo "  To use GPU acceleration: RUSTICL_ENABLE=panfrost python script.py"
    echo "  (automatically set when activating venv)"
else
    echo -e "  ${YELLOW}⚠${NC} GPU not detected - run 'sudo orangepi-config' to enable"
fi
echo ""
echo "To activate the environment manually:"
echo -e "  ${BLUE}source $VENV_DIR/bin/activate${NC}"
echo ""
echo "Quick start (GPU-accelerated):"
echo "  ./run_gpu.sh BTC 5m                        - Run GPU analysis directly"
echo "  ./run_web.sh                               - Standard web server"
echo "  ./run_web.sh parallel                      - Parallel web server (3.5x faster)"
echo "  ./run_web.sh vulkan                        - Vulkan web server"
echo ""
echo "Direct script usage:"
echo "  python gpu_wavelet_gpu_plot.py BTC 5m      - OpenCL GPU analysis"
echo "  python gpu_wavelet_gpu_console.py ETH 1h   - Console version with ASCII"
echo "  ./run_all_currencies.sh                    - Generate all 18 plots"
echo "  jupyter lab                                - Open notebooks"
echo ""
if [ ! -e /dev/dri/card0 ]; then
    echo -e "${YELLOW}NOTE:${NC} GPU not found. To enable Mali GPU:"
    echo "  1. sudo orangepi-config"
    echo "  2. System → Hardware → Enable gpu_mali"
    echo "  3. Reboot"
    echo "  4. Re-run this script or run: source ~/.bashrc"
    echo ""
    echo "Without GPU, scripts will show an error. Use --demo mode for CPU-only:"
    echo "  python gpu_wavelet_cpu_plot.py BTC 5m --demo"
    echo ""
fi
echo "======================================================================"
