#!/bin/bash
# Web Server Runner with GPU Support
# Usage: ./run_web.sh [parallel|standard|vulkan]

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

# Check GPU availability
GPU_STATUS="❌ Not Available"
if [ -e /dev/dri/card0 ]; then
    GPU_STATUS="✅ Mali-G31 (Panfrost)"
fi

# Determine which server to run
SERVER_TYPE="${1:-parallel}"

case "$SERVER_TYPE" in
    parallel)
        echo "======================================================================"
        echo "  Starting PARALLEL Web Server (Recommended)"
        echo "======================================================================"
        echo "  GPU Status: $GPU_STATUS"
        echo "  URL: http://localhost:8080"
        echo "  Features: Parallel plot generation (3.5x faster)"
        echo "======================================================================"
        echo ""
        python web_server_parallel.py
        ;;
    standard|sequential)
        echo "======================================================================"
        echo "  Starting STANDARD Web Server"
        echo "======================================================================"
        echo "  GPU Status: $GPU_STATUS"
        echo "  URL: http://localhost:8080"
        echo "  Features: Sequential plot generation"
        echo "======================================================================"
        echo ""
        python web_server.py
        ;;
    vulkan)
        echo "======================================================================"
        echo "  Starting VULKAN Web Server"
        echo "======================================================================"
        echo "  GPU Status: $GPU_STATUS"
        echo "  URL: http://localhost:8080"
        echo "  Features: Vulkan compute backend"
        echo "======================================================================"
        echo ""
        python web_server_vulkan.py
        ;;
    *)
        echo "Usage: ./run_web.sh [parallel|standard|vulkan]"
        echo ""
        echo "Options:"
        echo "  parallel   - Parallel plot generation (default, 3.5x faster)"
        echo "  standard   - Sequential plot generation"
        echo "  vulkan     - Vulkan compute backend"
        echo ""
        echo "Examples:"
        echo "  ./run_web.sh              # Start parallel server (default)"
        echo "  ./run_web.sh parallel     # Start parallel server"
        echo "  ./run_web.sh standard     # Start standard server"
        exit 1
        ;;
esac
