"""
CPU-Only Wavelet Decomposition (No OpenGL Dependencies)
========================================================

Pure NumPy implementation for Raspberry Pi Zero and systems without GPU support.
No OpenGL/GLFW imports - guaranteed to work on any system with numpy.

Usage:
    python gpu_wavelet_cpu_plot.py

Features:
- Pure NumPy convolution (no GPU dependencies)
- High-resolution plots (300 DPI)
- Real BTC data from Binance
- Works on Raspberry Pi Zero, Pi 1, 2, 3
- Saves plots to 'wavelet_plots_cpu/' directory
"""

import numpy as np
import time
import os
from datetime import datetime, timedelta

# Set matplotlib backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, safer for headless systems

import ccxt

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    print(f"⚠ Matplotlib import error: {e}")
    print("  Plots will not be generated")
    MATPLOTLIB_AVAILABLE = False

# Create output directory
output_dir = 'wavelet_plots_cpu'
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("WAVELET DECOMPOSITION - CPU MODE (NumPy)")
print("=" * 70)
print(f"\n  Pure CPU implementation (no OpenGL/GPU dependencies)")
if MATPLOTLIB_AVAILABLE:
    print(f"  Output Directory: {output_dir}/")
else:
    print(f"  ⚠ Matplotlib unavailable - plots disabled")
print()

# =============================================================================
# DEFINE WAVELETS
# =============================================================================

# Normalized wavelets (sum to 1.0 for price preservation)
haar_low_pass = np.array([0.5, 0.5], dtype=np.float32)
haar_high_pass = np.array([0.5, -0.5], dtype=np.float32)

db4_low_pass = np.array([
    0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551
], dtype=np.float32)
db4_sum = db4_low_pass.sum()
db4_low_pass = db4_low_pass / db4_sum

print("✓ Wavelets loaded (Haar and DB4)\n")

# =============================================================================
# CPU CONVOLUTION FUNCTION
# =============================================================================

def cpu_convolve(signal, filter_coeffs):
    """Pure NumPy convolution."""
    return np.convolve(signal, filter_coeffs, mode='valid').astype(np.float32)

# =============================================================================
# FETCH REAL BTC DATA
# =============================================================================

print("=" * 70)
print("FETCHING BTC DATA FROM BINANCE")
print("=" * 70)

try:
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = 'BTC/USDT'
    timeframe = '5m'
    
    # Calculate timestamp for 2 weeks back
    two_weeks_ago = datetime.now() - timedelta(weeks=2)
    since = int(two_weeks_ago.timestamp() * 1000)
    
    print(f"\n  Downloading {symbol} {timeframe} data from {two_weeks_ago.strftime('%Y-%m-%d %H:%M')}... ", end='', flush=True)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    print("✓")
    
    timestamps = np.array([candle[0] for candle in ohlcv])
    prices = np.array([candle[4] for candle in ohlcv], dtype=np.float32)
    volumes = np.array([candle[5] for candle in ohlcv], dtype=np.float32)
    dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]
    
    print(f"  Data range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Current BTC: ${prices[-1]:,.2f}")
    print(f"  Change: {((prices[-1] - prices[0]) / prices[0] * 100):+.2f}%")
    print(f"  Data points: {len(prices)}\n")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("  Generating synthetic data...")
    
    # Fallback to synthetic data
    np.random.seed(42)
    n_points = 1000
    initial_price = 50000.0
    drift = 0.0001
    volatility = 0.02
    returns = np.random.randn(n_points) * volatility + drift
    price_multipliers = np.exp(np.cumsum(returns))
    prices = (initial_price * price_multipliers).astype(np.float32)
    volumes = np.random.uniform(100, 1000, n_points).astype(np.float32)
    dates = [datetime.now() - timedelta(minutes=5*(n_points-i)) for i in range(n_points)]
    
    print(f"  Generated {n_points} synthetic price points\n")

# =============================================================================
# PERFORM WAVELET DECOMPOSITION
# =============================================================================

print("=" * 70)
print("COMPUTING WAVELET DECOMPOSITION")
print("=" * 70)

start = time.time()
trend_gpu = cpu_convolve(prices, haar_low_pass)
detail_gpu = cpu_convolve(prices, haar_high_pass)
trend_db4 = cpu_convolve(prices, db4_low_pass)
compute_time = time.time() - start

print(f"\n✓ Decomposition complete (CPU/NumPy): {compute_time*1000:.2f}ms")
print(f"  Trend points: {len(trend_gpu)}")
print(f"  Detail points: {len(detail_gpu)}\n")

# Multi-level decomposition
approximations = []
details = []
current_signal = prices

for i in range(5):
    approx = cpu_convolve(current_signal, haar_low_pass)
    detail = cpu_convolve(current_signal, haar_high_pass)
    approximations.append(approx)
    details.append(detail)
    current_signal = approx

levels = approximations

# =============================================================================
# ANOMALY DETECTION
# =============================================================================

detail_abs = np.abs(detail_gpu)
median = np.median(detail_abs)
mad = np.median(np.abs(detail_abs - median))
threshold = 3.0
anomaly_threshold = median + threshold * mad
anomaly_indices = np.where(detail_abs > anomaly_threshold)[0]

if not MATPLOTLIB_AVAILABLE:
    print("\n" + "=" * 70)
    print("SKIPPING PLOTS (matplotlib unavailable)")
    print("=" * 70)
    print("\n✓ Wavelet decomposition completed successfully")
    print(f"  Processing Time: {compute_time*1000:.2f}ms")
    print("=" * 70)
    exit(0)

print("=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# =============================================================================
# PLOT 1: MAIN OVERVIEW (4 SUBPLOTS)
# =============================================================================

print("\n[1/5] Main overview plot... ", end='', flush=True)

try:
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = GridSpec(4, 1, figure=fig, hspace=0.3)

    offset = len(prices) - len(trend_gpu)
    aligned_dates = dates[offset:]

    # Subplot 1: Original prices
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, prices, 'b-', linewidth=1, alpha=0.7, label='Original Price')
    ax1.set_title('BTC/USDT Price History (5-min candles)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Subplot 2: Price with Trend
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, prices, 'b-', linewidth=1, alpha=0.4, label='Original Price')
    ax2.plot(aligned_dates, trend_gpu, 'r-', linewidth=2, label='Trend (Low-pass)')
    ax2.set_title('Price Decomposition: Original vs Trend', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price (USD)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Subplot 3: Detail coefficients
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(aligned_dates, detail_gpu, 'g-', linewidth=1, alpha=0.7, label='Detail (High-pass)')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.axhline(y=anomaly_threshold, color='r', linestyle='--', linewidth=1, label=f'Anomaly Threshold (${anomaly_threshold:.0f})')
    ax3.axhline(y=-anomaly_threshold, color='r', linestyle='--', linewidth=1)
    if len(anomaly_indices) > 0:
        anomaly_dates = [aligned_dates[i] for i in anomaly_indices if i < len(aligned_dates)]
        anomaly_values = detail_gpu[anomaly_indices[:len(anomaly_dates)]]
        ax3.scatter(anomaly_dates, anomaly_values, color='red', s=50, marker='*', 
                   zorder=5, label=f'Anomalies ({len(anomaly_indices)})')
    ax3.set_title('Detail Coefficients (High-Frequency Changes)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Detail Coefficient (USD)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Subplot 4: Volume
    ax4 = fig.add_subplot(gs[3])
    ax4.bar(dates, volumes, width=0.04, color='purple', alpha=0.6)
    ax4.set_title('Trading Volume', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Volume (BTC)', fontsize=11)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_main_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# =============================================================================
# PLOT 2: PROGRESSIVE APPROXIMATIONS
# =============================================================================

print("[2/5] Progressive approximations... ", end='', flush=True)

try:
    fig = plt.figure(figsize=(20, 16), dpi=300)
    gs = GridSpec(6, 1, figure=fig, hspace=0.35)
    fig.suptitle('Progressive Approximations - CPU Mode', fontsize=18, fontweight='bold')

    ax_orig = plt.subplot(gs[0])
    ax_orig.plot(dates, prices, 'b-', linewidth=1.5, alpha=0.8, label='Original')
    ax_orig.set_title('Original Signal: BTC/USDT', fontsize=12, fontweight='bold')
    ax_orig.set_ylabel('Price (USD)', fontsize=10)
    ax_orig.grid(True, alpha=0.3)
    ax_orig.legend(loc='upper left')
    ax_orig.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    colors = ['purple', 'red', 'orange', 'green', 'blue']
    labels = ['Level 0 (Trend)', 'Level 1', 'Level 2', 'Level 3', 'Level 4']

    for i in range(5):
        ax = plt.subplot(gs[i+1])
        offset_a = len(prices) - len(approximations[i])
        dates_a = dates[offset_a:]
        ax.plot(dates_a, approximations[i], linewidth=2, color=colors[i], label=labels[i])
        ax.set_title(f'Approximation {labels[i]}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ax.set_xlabel('Date', fontsize=10)
    plt.savefig(f'{output_dir}/02_progressive_approximations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# =============================================================================
# PLOT 3: FREQUENCY BANDS
# =============================================================================

print("[3/5] Frequency bands... ", end='', flush=True)

try:
    fig = plt.figure(figsize=(20, 16), dpi=300)
    gs = GridSpec(6, 1, figure=fig, hspace=0.35)
    fig.suptitle('Frequency Band Decomposition', fontsize=18, fontweight='bold')

    ax_orig = plt.subplot(gs[0])
    ax_orig.plot(dates, prices, 'b-', linewidth=1.5, alpha=0.8)
    ax_orig.set_title('Original Signal', fontsize=12, fontweight='bold')
    ax_orig.set_ylabel('Price (USD)', fontsize=10)
    ax_orig.grid(True, alpha=0.3)
    ax_orig.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    detail_colors = ['red', 'orange', 'gold', 'green', 'blue']

    for i in range(5):
        ax = plt.subplot(gs[i+1])
        offset_d = len(prices) - len(details[i])
        dates_d = dates[offset_d:]
        ax.plot(dates_d, details[i], linewidth=1, color=detail_colors[i], alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.fill_between(dates_d, 0, details[i], alpha=0.3, color=detail_colors[i])
        ax.set_title(f'Detail Band {i+1}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Coefficient', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.savefig(f'{output_dir}/03_frequency_bands.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# =============================================================================
# PLOT 4: ANOMALY DETECTION
# =============================================================================

print("[4/5] Anomaly detection... ", end='', flush=True)

try:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=300)
    fig.suptitle('Anomaly Detection Analysis (CPU)', fontsize=16, fontweight='bold')

    ax1.plot(aligned_dates, detail_gpu, 'b-', linewidth=1, alpha=0.7, label='Detail Coefficients')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(y=anomaly_threshold, color='r', linestyle='--', linewidth=1.5, 
               label=f'Threshold: ${anomaly_threshold:.2f}')
    ax1.axhline(y=-anomaly_threshold, color='r', linestyle='--', linewidth=1.5)

    if len(anomaly_indices) > 0:
        anomaly_dates = [aligned_dates[i] for i in anomaly_indices if i < len(aligned_dates)]
        anomaly_values = detail_gpu[anomaly_indices[:len(anomaly_dates)]]
        ax1.scatter(anomaly_dates, anomaly_values, color='red', s=100, marker='*', 
                   zorder=5, edgecolors='darkred', linewidth=1, label=f'Anomalies: {len(anomaly_indices)}')

    ax1.set_title('Detail Coefficients with Anomaly Markers', fontsize=13)
    ax1.set_ylabel('Detail Coefficient (USD)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ax2.plot(aligned_dates, detail_abs, 'purple', linewidth=1, alpha=0.7, label='Absolute Detail')
    ax2.axhline(y=anomaly_threshold, color='r', linestyle='--', linewidth=1.5)
    ax2.fill_between(aligned_dates, 0, anomaly_threshold, alpha=0.2, color='green', label='Normal Range')
    ax2.fill_between(aligned_dates, anomaly_threshold, detail_abs.max(), alpha=0.2, color='red', label='Anomaly Zone')

    ax2.set_title('Volatility Measure', fontsize=13)
    ax2.set_ylabel('Absolute Detail (USD)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f'{output_dir}/04_anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# =============================================================================
# PLOT 5: TRADING SIGNALS
# =============================================================================

print("[5/5] Trading signals... ", end='', flush=True)

try:
    trend = levels[2]
    offset = len(prices) - len(trend)
    aligned_prices = prices[offset:]
    signal_dates = dates[offset:]
    price_deviation = aligned_prices - trend
    buffer = np.std(price_deviation) * 0.5

    buy_signals = price_deviation < -buffer
    sell_signals = price_deviation > buffer

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=300)
    fig.suptitle('Trading Signal Generation (CPU)', fontsize=16, fontweight='bold')

    ax1.plot(signal_dates, aligned_prices, 'b-', linewidth=1.5, alpha=0.7, label='Price')
    ax1.plot(signal_dates, trend, 'orange', linewidth=2, label='Trend (Level 3)')
    ax1.fill_between(signal_dates, trend - buffer, trend + buffer, alpha=0.2, color='gray', label='Neutral Zone')

    buy_dates = [signal_dates[i] for i in range(len(buy_signals)) if buy_signals[i]]
    buy_prices = aligned_prices[buy_signals]
    sell_dates = [signal_dates[i] for i in range(len(sell_signals)) if sell_signals[i]]
    sell_prices = aligned_prices[sell_signals]

    ax1.scatter(buy_dates, buy_prices, color='green', s=100, marker='^', 
               zorder=5, label=f'BUY ({buy_signals.sum()})', edgecolors='darkgreen', linewidth=1)
    ax1.scatter(sell_dates, sell_prices, color='red', s=100, marker='v', 
               zorder=5, label=f'SELL ({sell_signals.sum()})', edgecolors='darkred', linewidth=1)

    ax1.set_title('Price with Trading Signals', fontsize=13)
    ax1.set_ylabel('Price (USD)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ax2.plot(signal_dates, price_deviation, 'purple', linewidth=1.5, alpha=0.7, label='Price Deviation')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axhline(y=buffer, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Sell Threshold')
    ax2.axhline(y=-buffer, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Buy Threshold')
    ax2.fill_between(signal_dates, -buffer, buffer, alpha=0.2, color='gray')

    ax2.set_title('Deviation from Trend', fontsize=13)
    ax2.set_ylabel('Deviation (USD)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f'{output_dir}/05_trading_signals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\n✓ All plots saved to '{output_dir}/' directory")
print(f"\nGenerated files:")
print(f"  1. 01_main_overview.png          - 4-panel overview")
print(f"  2. 02_progressive_approximations.png - Multi-level filtering")
print(f"  3. 03_frequency_bands.png        - Detail coefficients")
print(f"  4. 04_anomaly_detection.png      - Anomaly analysis")
print(f"  5. 05_trading_signals.png        - Buy/sell signals")
print(f"\nProcessing Mode: CPU (NumPy)")
print(f"Processing Time: {compute_time*1000:.2f}ms")
print("=" * 70)
