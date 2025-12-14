#!/bin/bash

# Run GPU wavelet console analysis for all cryptocurrencies
# Generates ASCII console output with period and amplitude metrics for BTC, ETH, and SOL

echo "======================================================================"
echo "MULTI-CURRENCY WAVELET ANALYSIS - CONSOLE MODE"
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
    
    # Run the Python script with currency argument
    .venv/bin/python gpu_wavelet_gpu_console.py "$CURRENCY"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $CURRENCY analysis complete"
    else
        echo ""
        echo "✗ $CURRENCY analysis failed"
        exit 1
    fi
    
    # Small delay to avoid rate limiting
    if [ "$CURRENCY" != "SOL" ]; then
        echo ""
        echo "Waiting 2 seconds before next currency..."
        sleep 2
    fi
done

echo ""
echo "======================================================================"
echo "ALL CURRENCIES ANALYZED SUCCESSFULLY"
echo "======================================================================"
echo ""
echo "Results include:"
echo "  - 8-level wavelet decomposition"
echo "  - Period analysis (Min→Min, Max→Max)"
echo "  - Amplitude measurements"
echo "  - ASCII visualization graphs"
echo "======================================================================"
