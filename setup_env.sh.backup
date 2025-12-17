#!/bin/bash
# Trading Analytics Environment Setup Script
# ===========================================
# Creates virtual environment and installs dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  Trading Analytics - Environment Setup"
echo "======================================================================"
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}✗ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}[INFO]${NC} Python version: $PYTHON_VERSION"
echo ""

# Detect hardware platform
PLATFORM="unknown"
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "unknown")
    if [[ $MODEL == *"Raspberry Pi Zero"* ]]; then
        PLATFORM="pi-zero"
        echo -e "${YELLOW}[DETECTED]${NC} Raspberry Pi Zero"
    elif [[ $MODEL == *"Raspberry Pi 5"* ]]; then
        PLATFORM="pi-5"
        echo -e "${YELLOW}[DETECTED]${NC} Raspberry Pi 5"
    elif [[ $MODEL == *"Raspberry Pi"* ]]; then
        PLATFORM="pi-other"
        echo -e "${YELLOW}[DETECTED]${NC} Raspberry Pi (older model)"
    fi
fi

# Installation mode selection
echo ""
echo "Select installation mode:"
echo ""
echo "  1) Full     - All dependencies (notebooks, GPU acceleration)"
echo "  2) Minimal  - Core only (numpy, pandas, matplotlib)"
echo "  3) Pi Zero  - Ultra-minimal (numpy only, demo mode compatible)"
echo "  4) GPU      - Minimal + GPU acceleration (CUDA/OpenCL/OpenGL)"
echo ""

if [ "$PLATFORM" = "pi-zero" ]; then
    echo -e "${YELLOW}[RECOMMENDED]${NC} Option 3 (Pi Zero) - ccxt causes bus errors on Pi Zero"
    DEFAULT_MODE="3"
elif [ "$PLATFORM" = "pi-5" ]; then
    echo -e "${YELLOW}[RECOMMENDED]${NC} Option 4 (GPU) for OpenGL compute shaders"
    DEFAULT_MODE="4"
else
    echo -e "${YELLOW}[RECOMMENDED]${NC} Option 1 (Full) for complete functionality"
    DEFAULT_MODE="1"
fi

echo ""
read -p "Enter choice [1-4] (default: $DEFAULT_MODE): " MODE
MODE=${MODE:-$DEFAULT_MODE}

# Virtual environment directory
VENV_DIR="venv"

# Remove existing venv if it exists
if [ -d "$VENV_DIR" ]; then
    echo ""
    read -p "Virtual environment exists. Remove and recreate? [y/N]: " RECREATE
    if [[ $RECREATE =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}[INFO]${NC} Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo -e "${YELLOW}[SKIP]${NC} Using existing virtual environment"
        source "$VENV_DIR/bin/activate"
        MODE="upgrade"
    fi
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo -e "${BLUE}[STEP 1/3]${NC} Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}✓${NC} Virtual environment created: $VENV_DIR/"
fi

# Activate virtual environment
echo ""
echo -e "${BLUE}[STEP 2/3]${NC} Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Upgrade pip
echo ""
echo -e "${BLUE}[INFO]${NC} Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies based on mode
echo ""
echo -e "${BLUE}[STEP 3/3]${NC} Installing dependencies..."

case $MODE in
    1)
        # Full installation
        echo -e "${BLUE}[INFO]${NC} Full installation - all dependencies"
        pip install -r requirements.txt
        echo -e "${GREEN}✓${NC} All dependencies installed"
        ;;
    2)
        # Minimal installation
        echo -e "${BLUE}[INFO]${NC} Minimal installation"
        pip install numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 ccxt>=4.0.0
        echo -e "${GREEN}✓${NC} Core dependencies installed"
        ;;
    3)
        # Pi Zero safe mode
        echo -e "${BLUE}[INFO]${NC} Pi Zero safe installation (numpy only)"
        pip install numpy>=1.24.0
        echo -e "${GREEN}✓${NC} NumPy installed"
        echo ""
        echo -e "${YELLOW}[NOTE]${NC} Run scripts with: python gpu_wavelet_cpu_plot.py --demo --no-plots"
        ;;
    4)
        # GPU mode
        echo -e "${BLUE}[INFO]${NC} GPU installation"
        pip install numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 ccxt>=4.0.0
        
        echo ""
        echo "Select GPU backend:"
        echo "  1) CUDA   (NVIDIA GPUs)"
        echo "  2) OpenCL (Multi-platform)"
        echo "  3) OpenGL (Raspberry Pi 5)"
        read -p "Enter choice [1-3]: " GPU_MODE
        
        case $GPU_MODE in
            1)
                echo -e "${BLUE}[INFO]${NC} Installing PyTorch with CUDA..."
                pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0
                ;;
            2)
                echo -e "${BLUE}[INFO]${NC} Installing PyOpenCL..."
                pip install pyopencl>=2023.1.0
                ;;
            3)
                echo -e "${BLUE}[INFO]${NC} Installing OpenGL support..."
                pip install PyOpenGL>=3.1.7 glfw>=2.6.0
                ;;
        esac
        echo -e "${GREEN}✓${NC} GPU dependencies installed"
        ;;
    upgrade)
        # Upgrade existing installation
        echo -e "${BLUE}[INFO]${NC} Upgrading existing packages..."
        pip install --upgrade -r requirements.txt
        echo -e "${GREEN}✓${NC} Packages upgraded"
        ;;
    *)
        echo -e "${RED}✗ Invalid mode selected${NC}"
        exit 1
        ;;
esac

# Summary
echo ""
echo "======================================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "======================================================================"
echo ""
echo "Virtual environment: $VENV_DIR/"
echo ""
echo "To activate the environment:"
echo -e "  ${BLUE}source $VENV_DIR/bin/activate${NC}"
echo ""
echo "Available scripts:"
echo "  • btc-prediction.ipynb           - LSTM prediction notebook"
echo "  • wave_nada.ipynb                - Wavelet analysis notebook"
echo "  • gpu_wavelet_cuda_plot.py       - CUDA acceleration"
echo "  • gpu_wavelet_gpu_plot.py        - OpenCL acceleration"
echo "  • gpu_wavelet_opengl_plot.py     - OpenGL acceleration (Pi 5)"
echo "  • gpu_wavelet_cpu_plot.py        - CPU-only (Pi Zero safe with --demo)"
echo ""

if [ "$MODE" = "3" ] || [ "$PLATFORM" = "pi-zero" ]; then
    echo -e "${YELLOW}[Pi Zero Usage]${NC}"
    echo "  python gpu_wavelet_cpu_plot.py --demo --no-plots"
    echo ""
fi

echo "======================================================================"
