#!/bin/bash
# Run GPU console analysis with environment configured
export RUSTICL_ENABLE=panfrost

CURRENCY="${1:-BTC}"
TIMEFRAME="${2:-5m}"

echo "Running GPU console analysis for $CURRENCY/$TIMEFRAME..."
.venv/bin/python gpu_wavelet_gpu_console.py "$CURRENCY" "$TIMEFRAME"
