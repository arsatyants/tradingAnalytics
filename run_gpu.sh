#!/bin/bash
# GPU Script Runner with Correct Environment Variables
# Usage: ./run_gpu.sh <script_name> [args...]
# Example: ./run_gpu.sh gpu_wavelet_gpu_plot.py BTC 5m

set -e

# Set GPU environment variables
export RUSTICL_ENABLE=panfrost

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Run ./setup_env.sh first."
    exit 1
fi

# Check if GPU is available
if [ ! -e /dev/dri/card0 ]; then
    echo "⚠️  WARNING: GPU device not found at /dev/dri/card0"
    echo "   Run 'sudo orangepi-config' to enable gpu_mali"
    echo ""
fi

# Run the script with all arguments
if [ $# -eq 0 ]; then
    echo "Usage: ./run_gpu.sh <script_name> [args...]"
    echo ""
    echo "Examples:"
    echo "  ./run_gpu.sh gpu_wavelet_gpu_plot.py BTC 5m"
    echo "  ./run_gpu.sh gpu_wavelet_gpu_console.py ETH 1h"
    echo "  ./run_gpu.sh web_server_parallel.py"
    exit 1
fi

echo "Running: $@ with GPU enabled (Mali-G31 Panfrost)"
echo ""
python "$@"
