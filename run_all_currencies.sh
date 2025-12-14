#!/bin/bash

# Run GPU wavelet analysis for all cryptocurrencies
# Generates separate plots for BTC, ETH, and SOL
# Usage: ./run_all_currencies.sh [timeframe]
# Example: ./run_all_currencies.sh 1h

TIMEFRAME="${1:-5m}"  # Default to 5m if not specified

echo "======================================================================"
echo "MULTI-CURRENCY WAVELET ANALYSIS (Timeframe: $TIMEFRAME)"
echo "======================================================================"
echo ""

# Array of currencies to process
CURRENCIES=("BTC" "ETH" "SOL")

# Process each currency
for CURRENCY in "${CURRENCIES[@]}"; do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "Processing: $CURRENCY"
    echo "----------------------------------------------------------------------"
    
    # Run the Python script with currency and timeframe arguments
    .venv/bin/python gpu_wavelet_gpu_plot.py "$CURRENCY" "$TIMEFRAME"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ $CURRENCY processing complete"
    else
        echo "✗ $CURRENCY processing failed"
        exit 1
    fi
    
    # Small delay to avoid rate limiting
    sleep 2
done

echo ""
echo "======================================================================"
echo "ALL CURRENCIES PROCESSED SUCCESSFULLY"
echo "======================================================================"
echo ""
echo "Generated plots:"
echo "  - wavelet_plots/btc/  (6 PNG files)"
echo "  - wavelet_plots/eth/  (6 PNG files)"
echo "  - wavelet_plots/sol/  (6 PNG files)"
echo ""
echo "Total: 18 high-resolution plots (300 DPI)"
echo "======================================================================"
