#!/bin/bash
# Trading Analytics Environment Setup Script
# ===========================================
# Automatic setup: Creates .venv and installs all dependencies

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  Trading Analytics - Automated Environment Setup"
echo "======================================================================"
echo ""

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
echo -e "${BLUE}[1/4]${NC} Creating virtual environment..."
$PYTHON_CMD -m venv "$VENV_DIR"
echo -e "${GREEN}✓${NC} Created: $VENV_DIR/"

# Activate virtual environment
echo ""
echo -e "${BLUE}[2/4]${NC} Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓${NC} Activated"

# Upgrade pip
echo ""
echo -e "${BLUE}[3/4]${NC} Upgrading pip..."
pip install --upgrade pip -q
echo -e "${GREEN}✓${NC} pip upgraded"

# Install all dependencies
echo ""
echo -e "${BLUE}[4/4]${NC} Installing all dependencies from requirements.txt..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo ""
echo -e "${GREEN}✓${NC} All dependencies installed"

# Summary
echo ""
echo "======================================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "======================================================================"
echo ""
echo "Virtual environment: $VENV_DIR/"
echo ""
echo "To activate the environment manually:"
echo -e "  ${BLUE}source $VENV_DIR/bin/activate${NC}"
echo ""
echo "Quick start:"
echo "  ./run_all_currencies.sh          - Generate plots for BTC/ETH/SOL"
echo "  python web_server.py             - Launch web interface at :8080"
echo "  jupyter lab                      - Open notebooks"
echo ""
echo "======================================================================"
