#!/bin/bash
# Trading Analytics Environment Setup Script
# ===========================================
# Automatic setup: Creates .venv and installs all dependencies + GPU support

# Don't exit on error for optional components
set +e

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
echo -e "${BLUE}[3/7]${NC} Upgrading pip..."
pip install --upgrade pip -q
echo -e "${GREEN}✓${NC} pip upgraded"

# Install system dependencies for Python packages
echo ""
echo -e "${BLUE}[4/7]${NC} Installing system dependencies..."
echo "This requires sudo privileges..."

# Python development headers (required for pyopencl)
sudo apt update -qq
sudo apt install -y python3-dev python3-pip build-essential cmake ocl-icd-opencl-dev clinfo >/dev/null 2>&1 || echo -e "${YELLOW}⚠${NC} Some system packages may not be available"
echo -e "${GREEN}✓${NC} System dependencies installed"

# Install core dependencies (without pyopencl)
echo ""
echo -e "${BLUE}[5/7]${NC} Installing core Python dependencies..."
echo "This may take several minutes..."

# Create temporary requirements without pyopencl
grep -v "^pyopencl" requirements.txt > /tmp/requirements_core.txt

pip install -r /tmp/requirements_core.txt
echo ""
echo -e "${GREEN}✓${NC} Core dependencies installed"

# Try to install pyopencl (optional)
echo ""
echo -e "${BLUE}[6/7]${NC} Installing GPU support (pyopencl - optional)..."
if pip install pyopencl==2025.2.7 2>&1 | tee /tmp/pyopencl_install.log; then
    echo -e "${GREEN}✓${NC} pyopencl installed successfully"
else
    echo -e "${YELLOW}⚠${NC} pyopencl installation failed (GPU acceleration disabled)"
    echo "    CPU fallback will be used. Check /tmp/pyopencl_install.log for details."
    echo "    The project will still work with gpu_wavelet_cpu_plot.py"
fi

# GPU Runtime Setup (Vulkan & Mesa)
echo ""
echo -e "${BLUE}[7/7]${NC} Setting up GPU runtime (Vulkan & Mesa)..."
echo ""

# Install Vulkan support (optional)
echo "Installing Vulkan runtime (optional)..."
sudo apt install -y vulkan-tools mesa-vulkan-drivers libvulkan1 >/dev/null 2>&1 || echo -e "${YELLOW}⚠${NC} Vulkan packages not available"
echo -e "${GREEN}✓${NC} Vulkan setup complete"

# Install Mesa utilities (optional)
echo "Installing Mesa GPU utilities (optional)..."
sudo apt install -y mesa-utils libgl1-mesa-dri mesa-opencl-icd >/dev/null 2>&1 || echo -e "${YELLOW}⚠${NC} Mesa packages not available"
echo -e "${GREEN}✓${NC} Mesa utilities installed"

# Load panfrost GPU driver (Orange Pi specific)
echo "Loading panfrost GPU driver (Orange Pi specific)..."
sudo modprobe panfrost 2>/dev/null || echo -e "${YELLOW}⚠${NC} Panfrost module not loaded (enable GPU in orangepi-config if on Orange Pi)"
echo "panfrost" | sudo tee /etc/modules-load.d/panfrost.conf >/dev/null 2>&1 || true
echo -e "${GREEN}✓${NC} Panfrost configured"

# Set GPU permissions
echo "Configuring GPU permissions..."
sudo usermod -a -G render $USER 2>/dev/null || true
sudo usermod -a -G video $USER 2>/dev/null || true
echo -e "${GREEN}✓${NC} User added to render and video groups"

# Configure environment
echo ""
echo "Configuring GPU environment variables..."

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

# Check what was installed
PYOPENCL_INSTALLED=false
$VENV_DIR/bin/python -c "import pyopencl" 2>/dev/null && PYOPENCL_INSTALLED=true

# Summary
echo ""
echo "======================================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "======================================================================"
echo ""
echo "Virtual environment: $VENV_DIR/"
echo ""
echo "Python Packages:"
echo -e "  ${GREEN}✓${NC} Core dependencies numpy, pandas, torch, etc."
if [ "$PYOPENCL_INSTALLED" = true ]; then
    echo -e "  ${GREEN}✓${NC} pyopencl - GPU acceleration via OpenCL"
else
    echo -e "  ${YELLOW}⚠${NC} pyopencl not installed - CPU fallback available"
fi
echo ""
echo "GPU Status:"
if [ -e /dev/dri/card0 ]; then
    echo -e "  ${GREEN}✓${NC} GPU device detected: /dev/dri/card0"
    if [ "$PYOPENCL_INSTALLED" = true ]; then
        echo -e "  ${GREEN}✓${NC} OpenCL GPU acceleration available"
        echo "  To use GPU: RUSTICL_ENABLE=panfrost python script.py"
        echo "  (automatically set when activating venv)"
    else
        echo -e "  ${YELLOW}⚠${NC} GPU detected but pyopencl not installed"
        echo "  Run: source .venv/bin/activate && pip install pyopencl"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} GPU device not detected"
    if [[ "$MODEL" == *"OrangePi"* ]]; then
        echo "  Enable GPU in orangepi-config and reboot"
    fi
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
