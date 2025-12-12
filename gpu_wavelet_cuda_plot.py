"""
GPU-Accelerated Wavelet Decomposition with CUDA (PyTorch)
==========================================================

This script uses CUDA via PyTorch for GPU acceleration instead of OpenCL.
Features:
- CUDA-accelerated wavelet computations via PyTorch
- High-resolution plots (300 DPI)
- Multiple subplots for comprehensive analysis
- Real BTC data from Binance
- Automatic fallback to CPU if CUDA unavailable
- Saves all plots to 'wavelet_plots_cuda/' directory
"""

import torch
import numpy as np
import time
import ccxt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import os

# Create output directory
output_dir = 'wavelet_plots_cuda'
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("GPU-ACCELERATED WAVELET DECOMPOSITION - CUDA MODE")
print("=" * 70)

# =============================================================================
# STEP 1: INITIALIZE CUDA
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"\n✓ CUDA Initialized: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
else:
    print(f"\n⚠ CUDA not available, using CPU")

print(f"  Device: {device}")
print(f"  Output Directory: {output_dir}/\n")

# =============================================================================
# STEP 2: DEFINE WAVELETS
# =============================================================================

# Normalized wavelets (sum to 1.0 for price preservation)
haar_low_pass = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
haar_high_pass = torch.tensor([0.5, -0.5], dtype=torch.float32, device=device)

db4_low_pass = torch.tensor([
    0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551
], dtype=torch.float32, device=device)
db4_sum = db4_low_pass.sum()
db4_low_pass = db4_low_pass / db4_sum

print("✓ Wavelets loaded (Haar and DB4)\n")

# =============================================================================
# STEP 3: CUDA CONVOLUTION FUNCTION
# =============================================================================

def cuda_convolve(signal, filter_coeffs):
    """
    Perform 1D convolution using PyTorch's optimized CUDA kernels
    
    PyTorch uses highly optimized cuDNN/cuBLAS libraries for convolution,
    which are much faster than custom CUDA kernels for most use cases.
    """
    # Ensure inputs are on the correct device
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32, device=device)
    if not isinstance(filter_coeffs, torch.Tensor):
        filter_coeffs = torch.tensor(filter_coeffs, dtype=torch.float32, device=device)
    
    # Reshape for conv1d: (batch, channels, length)
    signal = signal.view(1, 1, -1)
    filter_coeffs = filter_coeffs.view(1, 1, -1)
    
    # Perform convolution using PyTorch's optimized CUDA implementation
    output = torch.nn.functional.conv1d(signal, filter_coeffs, padding='valid')
    
    # Return as 1D array
    return output.squeeze()

# =============================================================================
# STEP 4: FETCH REAL BTC DATA
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
    since = int(two_weeks_ago.timestamp() * 1000)  # Convert to milliseconds
    
    print(f"\n  Downloading {symbol} {timeframe} data from {two_weeks_ago.strftime('%Y-%m-%d %H:%M')}... ", end='', flush=True)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    print("✓")
    
    timestamps = np.array([candle[0] for candle in ohlcv])
    prices_np = np.array([candle[4] for candle in ohlcv], dtype=np.float32)
    volumes_np = np.array([candle[5] for candle in ohlcv], dtype=np.float32)
    dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]
    
    # Convert to PyTorch tensors on GPU
    prices = torch.tensor(prices_np, dtype=torch.float32, device=device)
    volumes = torch.tensor(volumes_np, dtype=torch.float32, device=device)
    
    print(f"  Data range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Current BTC: ${prices_np[-1]:,.2f}")
    print(f"  Change: {((prices_np[-1] - prices_np[0]) / prices_np[0] * 100):+.2f}%")
    print(f"  Data points: {len(prices)}\n")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    exit(1)

# =============================================================================
# STEP 5: PERFORM WAVELET DECOMPOSITION
# =============================================================================

print("=" * 70)
print("COMPUTING WAVELET DECOMPOSITION")
print("=" * 70)

# Synchronize GPU before timing
if torch.cuda.is_available():
    torch.cuda.synchronize()

start = time.time()
trend_cuda = cuda_convolve(prices, haar_low_pass)
detail_cuda = cuda_convolve(prices, haar_high_pass)
trend_db4 = cuda_convolve(prices, db4_low_pass)

# Synchronize GPU after computation
if torch.cuda.is_available():
    torch.cuda.synchronize()

cuda_time = time.time() - start

print(f"\n✓ Decomposition complete: {cuda_time*1000:.2f}ms")
print(f"  Trend points: {len(trend_cuda)}")
print(f"  Detail points: {len(detail_cuda)}\n")

# Multi-level decomposition (proper wavelet pyramid)
# Each level shows: approximation (trend) and detail at that scale
approximations = []  # Low-pass (smoothed trend)
details = []         # High-pass (detail at each scale)

current_signal = prices

for i in range(5):
    # Apply both low-pass and high-pass filters
    approx = cuda_convolve(current_signal, haar_low_pass)
    detail = cuda_convolve(current_signal, haar_high_pass)
    
    # Move to CPU for storage (keep GPU memory free)
    approximations.append(approx.cpu().numpy())
    details.append(detail.cpu().numpy())
    
    # Next level operates on the approximation (coarser scale)
    current_signal = approx

# Convert main outputs to NumPy for plotting
trend_gpu = trend_cuda.cpu().numpy()
detail_gpu = detail_cuda.cpu().numpy()
prices_np = prices.cpu().numpy()
volumes_np = volumes.cpu().numpy()

# =============================================================================
# STEP 6: ANOMALY DETECTION
# =============================================================================

detail_abs = np.abs(detail_gpu)
median = np.median(detail_abs)
mad = np.median(np.abs(detail_abs - median))
threshold = 3.0
anomaly_threshold = median + threshold * mad
anomaly_indices = np.where(detail_abs > anomaly_threshold)[0]

print("=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# =============================================================================
# PLOT 1: MAIN OVERVIEW (4 SUBPLOTS)
# =============================================================================

print("[1/6] Main overview... ", end='', flush=True)

fig = plt.figure(figsize=(18, 12), dpi=300)
gs = GridSpec(4, 1, figure=fig, hspace=0.3)

# Subplot 1: Original BTC Price
ax1 = plt.subplot(gs[0])
ax1.plot(dates, prices_np, 'b-', linewidth=1.5, label='BTC/USDT')
ax1.set_title(f'Original BTC Price - {timeframe} candles', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price (USD)', fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

# Subplot 2: Trend (Low-Pass Filter)
ax2 = plt.subplot(gs[1])
offset_t = len(prices_np) - len(trend_gpu)
dates_trend = dates[offset_t:]
ax2.plot(dates_trend, trend_gpu, 'g-', linewidth=2, label='Trend (Haar LP)')
ax2.set_title('Trend Component (Low-Pass Filtered)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Price (USD)', fontsize=11)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

# Subplot 3: Detail (High-Pass Filter)
ax3 = plt.subplot(gs[2])
offset_d = len(prices_np) - len(detail_gpu)
dates_detail = dates[offset_d:]
ax3.plot(dates_detail, detail_gpu, 'r-', linewidth=1, alpha=0.7, label='Detail (Haar HP)')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Mark anomalies
if len(anomaly_indices) > 0:
    anomaly_dates = [dates_detail[i] for i in anomaly_indices]
    anomaly_values = [detail_gpu[i] for i in anomaly_indices]
    ax3.scatter(anomaly_dates, anomaly_values, color='red', s=50, marker='*',
               label=f'Anomalies ({len(anomaly_indices)})', zorder=5)

ax3.set_title('Detail Component (High-Pass Filtered)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Detail Coefficient', fontsize=11)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

# Subplot 4: Volume
ax4 = plt.subplot(gs[3])
ax4.bar(dates, volumes_np, width=0.002, color='orange', alpha=0.6, label='Volume')
ax4.set_title('Trading Volume', fontsize=14, fontweight='bold')
ax4.set_ylabel('Volume (BTC)', fontsize=11)
ax4.set_xlabel('Date', fontsize=11)
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3, axis='y')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

plt.savefig(f'{output_dir}/01_main_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 2A: PROGRESSIVE APPROXIMATIONS
# =============================================================================

print("[2/6] Progressive approximations... ", end='', flush=True)

fig = plt.figure(figsize=(24, 20), dpi=300)
gs = GridSpec(6, 1, figure=fig, hspace=0.35)
fig.suptitle('Progressive Approximations - Frequency Filtering', fontsize=20, fontweight='bold')

# Original signal at top
ax_orig = plt.subplot(gs[0])
ax_orig.plot(dates, prices_np, 'b-', linewidth=1.5, alpha=0.8, label='Original (all frequencies)')
ax_orig.set_title('Original Signal: BTC/USDT', fontsize=13, fontweight='bold')
ax_orig.set_ylabel('Price (USD)', fontsize=11)
ax_orig.grid(True, alpha=0.3)
ax_orig.legend(loc='upper left')
ax_orig.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

# Frequency bands explanation
freq_bands = [
    ('1-2 intervals', '~50% of frequencies', 'Noise & very short-term fluctuations'),
    ('2-4 intervals', '~25% of frequencies', 'Short-term trading movements'),
    ('4-8 intervals', '~12.5% of frequencies', 'Intraday trend changes'),
    ('8-16 intervals', '~6.25% of frequencies', 'Daily cycle patterns'),
    ('16+ intervals', '~3.125% of frequencies', 'Multi-day trend reversals')
]

# Plot progressive approximations with overlays showing differences
approx_colors = ['orangered', 'orange', 'gold', 'limegreen', 'dodgerblue']
approx_labels = [
    'After removing 1-2 noise',
    'After removing up to 4 fluctuations',
    'After removing up to 8 movements',
    'After removing up to 16 variations',
    'Main trend only (16+ timescale)'
]

for i in range(5):
    ax_approx = plt.subplot(gs[i+1])
    
    offset_a = len(prices_np) - len(approximations[i])
    dates_a = dates[offset_a:]
    current_approx = approximations[i]
    
    # Plot previous level (if exists) to show what's being removed
    if i > 0:
        offset_prev = len(prices_np) - len(approximations[i-1])
        dates_prev = dates[offset_prev:]
        prev_approx = approximations[i-1]
        
        # Resample previous to match current length
        if len(prev_approx) != len(current_approx):
            # Simple downsampling by taking every other point
            indices = np.linspace(0, len(prev_approx)-1, len(current_approx)).astype(int)
            prev_approx_resampled = prev_approx[indices]
        else:
            prev_approx_resampled = prev_approx
            
        ax_approx.plot(dates_a, prev_approx_resampled, 'gray', linewidth=1.5, 
                      alpha=0.4, linestyle='--', label=f'Level {i} (previous)')
        
        # Show the difference (what was removed)
        difference = prev_approx_resampled - current_approx
        ax_diff = ax_approx.twinx()
        ax_diff.fill_between(dates_a, 0, difference, alpha=0.2, color='red', label='Removed')
        ax_diff.set_ylabel('Removed (USD)', fontsize=9, color='red')
        ax_diff.tick_params(axis='y', labelcolor='red', labelsize=8)
        ax_diff.set_ylim(-difference.std()*3, difference.std()*3)
    else:
        # First level - compare to original
        original_segment = prices_np[offset_a:offset_a+len(approximations[i])]
        ax_approx.plot(dates_a, original_segment, 'gray', linewidth=1.5, 
                      alpha=0.4, linestyle='--', label='Original')
        
        difference = original_segment - current_approx
        ax_diff = ax_approx.twinx()
        ax_diff.fill_between(dates_a, 0, difference, alpha=0.2, color='red')
        ax_diff.set_ylabel('Removed (USD)', fontsize=9, color='red')
        ax_diff.tick_params(axis='y', labelcolor='red', labelsize=8)
        ax_diff.set_ylim(-difference.std()*3, difference.std()*3)
    
    # Plot current approximation (main signal)
    ax_approx.plot(dates_a, current_approx, linewidth=2.5, alpha=0.9, 
                  color=approx_colors[i], label=f'Level {i+1}', zorder=10)
    
    # Calculate metrics
    if i > 0:
        smoothness_increase = (prev_approx_resampled.std() / (current_approx.std() + 1e-10)) 
        removed_variance = difference.std()
    else:
        original_segment = prices_np[offset_a:offset_a+len(approximations[i])]
        smoothness_increase = original_segment.std() / (current_approx.std() + 1e-10)
        removed_variance = difference.std()
    
    ax_approx.set_title(f'Level {i+1}: {approx_labels[i]}', 
                       fontsize=12, fontweight='bold', pad=10)
    ax_approx.set_ylabel('Price (USD)', fontsize=10)
    ax_approx.grid(True, alpha=0.3)
    ax_approx.legend(loc='upper left', fontsize=8)
    ax_approx.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    
    # Stats box showing what changed
    info_text = f'Smoothness: {smoothness_increase:.2f}x\nRemoved σ: ${removed_variance:.1f}\nFreq kept: ≥{freq_bands[i][0]}'
    ax_approx.text(0.98, 0.97, info_text, 
                  transform=ax_approx.transAxes, fontsize=8, 
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Color-coded border
    for spine in ax_approx.spines.values():
        spine.set_edgecolor(approx_colors[i])
        spine.set_linewidth(2.5)

ax_approx.set_xlabel('Date', fontsize=11)

plt.savefig(f'{output_dir}/02a_progressive_approximations.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 2B: FREQUENCY BANDS (DETAILS)
# =============================================================================

print("[3/6] Frequency bands (details)... ", end='', flush=True)

fig = plt.figure(figsize=(24, 20), dpi=300)
gs = GridSpec(6, 1, figure=fig, hspace=0.35)
fig.suptitle('Frequency Band Decomposition - Detail Coefficients', fontsize=20, fontweight='bold')

# Original signal at top
ax_orig = plt.subplot(gs[0])
ax_orig.plot(dates, prices_np, 'b-', linewidth=1.5, alpha=0.8)
ax_orig.set_title('Original Signal: BTC/USDT (All Frequencies Combined)', fontsize=13, fontweight='bold')
ax_orig.set_ylabel('Price (USD)', fontsize=11)
ax_orig.grid(True, alpha=0.3)
ax_orig.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
price_range = prices_np.max() - prices_np.min()
ax_orig.text(0.02, 0.95, f'Range: ${price_range:,.0f}\nStd: ${prices_np.std():,.0f}', 
            transform=ax_orig.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot each detail band
colors_map = ['red', 'orange', 'gold', 'green', 'blue']

for i in range(5):
    ax_detail = plt.subplot(gs[i+1])
    
    offset_d = len(prices_np) - len(details[i])
    dates_d = dates[offset_d:]
    
    # Plot detail coefficients with filled area
    ax_detail.plot(dates_d, details[i], linewidth=1.5, alpha=0.8, color=colors_map[i])
    ax_detail.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    ax_detail.fill_between(dates_d, 0, details[i], alpha=0.3, color=colors_map[i])
    
    # Statistics
    detail_std = details[i].std()
    detail_range = details[i].max() - details[i].min()
    detail_mean = abs(details[i]).mean()
    
    # Title with comprehensive info
    ax_detail.set_title(f'Band {i+1}: {freq_bands[i][0]} - {freq_bands[i][2]}', 
                       fontsize=12, fontweight='bold', pad=10)
    ax_detail.set_ylabel('Detail Coeff (USD)', fontsize=10)
    ax_detail.grid(True, alpha=0.3)
    ax_detail.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    
    # Stats box
    ax_detail.text(0.02, 0.97, 
                  f'Std: ${detail_std:.1f}\nMean|Δ|: ${detail_mean:.1f}\nRange: ${detail_range:.1f}\n{freq_bands[i][1]}', 
                  transform=ax_detail.transAxes, fontsize=8, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Color-coded border
    for spine in ax_detail.spines.values():
        spine.set_edgecolor(colors_map[i])
        spine.set_linewidth(2.5)

ax_detail.set_xlabel('Date', fontsize=11)

plt.savefig(f'{output_dir}/02b_frequency_bands.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 3: ANOMALY DETECTION
# =============================================================================

print("[4/6] Anomaly detection... ", end='', flush=True)

fig = plt.figure(figsize=(18, 10), dpi=300)
gs = GridSpec(2, 1, figure=fig, hspace=0.3)

# Subplot 1: Detail with anomalies highlighted
ax1 = plt.subplot(gs[0])
ax1.plot(dates_detail, detail_gpu, 'b-', linewidth=1, alpha=0.6, label='Detail')
ax1.axhline(y=anomaly_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (3σ MAD)')
ax1.axhline(y=-anomaly_threshold, color='red', linestyle='--', linewidth=2)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

if len(anomaly_indices) > 0:
    ax1.scatter(anomaly_dates, anomaly_values, color='red', s=100, marker='*',
               label=f'Anomalies ({len(anomaly_indices)})', zorder=5)

ax1.set_title('Anomaly Detection in Detail Coefficients', fontsize=14, fontweight='bold')
ax1.set_ylabel('Detail Coefficient', fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

# Subplot 2: Absolute volatility
ax2 = plt.subplot(gs[1])
ax2.plot(dates_detail, detail_abs, 'purple', linewidth=1.5, alpha=0.7, label='|Detail|')
ax2.axhline(y=anomaly_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
ax2.fill_between(dates_detail, 0, detail_abs, alpha=0.2, color='purple')

ax2.set_title('Absolute Volatility', fontsize=14, fontweight='bold')
ax2.set_ylabel('|Detail|', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

plt.savefig(f'{output_dir}/03_anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 4: TRADING SIGNALS
# =============================================================================

print("[5/6] Trading signals... ", end='', flush=True)

fig = plt.figure(figsize=(18, 10), dpi=300)
gs = GridSpec(2, 1, figure=fig, hspace=0.3)

# Calculate deviation from trend
deviation = prices_np[offset_t:] - trend_gpu
buy_signals = np.where(deviation < -deviation.std())[0]
sell_signals = np.where(deviation > deviation.std())[0]

# Subplot 1: Price with buy/sell markers
ax1 = plt.subplot(gs[0])
ax1.plot(dates_trend, prices_np[offset_t:], 'b-', linewidth=1.5, alpha=0.6, label='Price')
ax1.plot(dates_trend, trend_gpu, 'g-', linewidth=2, label='Trend')

if len(buy_signals) > 0:
    buy_dates = [dates_trend[i] for i in buy_signals]
    buy_prices = [prices_np[offset_t + i] for i in buy_signals]
    ax1.scatter(buy_dates, buy_prices, color='green', s=100, marker='^',
               label=f'Buy Signals ({len(buy_signals)})', zorder=5)

if len(sell_signals) > 0:
    sell_dates = [dates_trend[i] for i in sell_signals]
    sell_prices = [prices_np[offset_t + i] for i in sell_signals]
    ax1.scatter(sell_dates, sell_prices, color='red', s=100, marker='v',
               label=f'Sell Signals ({len(sell_signals)})', zorder=5)

ax1.set_title('Trading Signals Based on Trend Deviation', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price (USD)', fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

# Subplot 2: Deviation from trend
ax2 = plt.subplot(gs[1])
ax2.plot(dates_trend, deviation, 'purple', linewidth=1.5, alpha=0.7, label='Deviation')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axhline(y=deviation.std(), color='red', linestyle='--', linewidth=1, label='±1σ')
ax2.axhline(y=-deviation.std(), color='red', linestyle='--', linewidth=1)
ax2.fill_between(dates_trend, 0, deviation, alpha=0.2, color='purple')

ax2.set_title('Price Deviation from Trend', fontsize=14, fontweight='bold')
ax2.set_ylabel('Deviation (USD)', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

plt.savefig(f'{output_dir}/04_trading_signals.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 5: STATISTICS DASHBOARD
# =============================================================================

print("[6/6] Statistics dashboard... ", end='', flush=True)

fig = plt.figure(figsize=(18, 10), dpi=300)
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Histogram of price changes
ax1 = plt.subplot(gs[0, 0])
price_changes = np.diff(prices_np)
ax1.hist(price_changes, bins=50, color='blue', alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax1.set_title('Distribution of Price Changes', fontsize=12, fontweight='bold')
ax1.set_xlabel('Price Change (USD)', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Histogram of detail coefficients
ax2 = plt.subplot(gs[0, 1])
ax2.hist(detail_gpu, bins=50, color='red', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax2.set_title('Distribution of Detail Coefficients', fontsize=12, fontweight='bold')
ax2.set_xlabel('Detail Value', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Rolling volatility
ax3 = plt.subplot(gs[0, 2])
window = 20
rolling_std = np.array([np.std(prices_np[max(0, i-window):i+1]) for i in range(len(prices_np))])
ax3.plot(dates, rolling_std, 'purple', linewidth=1.5)
ax3.set_title(f'Rolling Volatility (window={window})', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Std Dev (USD)', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Statistics table
ax4 = plt.subplot(gs[1, :])
ax4.axis('off')

stats_data = [
    ['Metric', 'Value'],
    ['─' * 30, '─' * 30],
    ['Price Mean', f'${prices_np.mean():,.2f}'],
    ['Price Std Dev', f'${prices_np.std():,.2f}'],
    ['Price Min', f'${prices_np.min():,.2f}'],
    ['Price Max', f'${prices_np.max():,.2f}'],
    ['Price Range', f'${prices_np.max() - prices_np.min():,.2f}'],
    ['─' * 30, '─' * 30],
    ['Detail Mean', f'{detail_gpu.mean():.4f}'],
    ['Detail Std Dev', f'{detail_gpu.std():.2f}'],
    ['Detail Min', f'{detail_gpu.min():.2f}'],
    ['Detail Max', f'{detail_gpu.max():.2f}'],
    ['─' * 30, '─' * 30],
    ['Anomalies Detected', f'{len(anomaly_indices)}'],
    ['Anomaly Threshold', f'±{anomaly_threshold:.2f}'],
    ['Buy Signals', f'{len(buy_signals)}'],
    ['Sell Signals', f'{len(sell_signals)}'],
]

table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                 colWidths=[0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)

plt.savefig(f'{output_dir}/05_statistics_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)

print(f"\nGenerated files:")
print(f"  1. 01_main_overview.png               - 4-panel overview")
print(f"  2. 02a_progressive_approximations.png - Progressive filtering (accumulations)")
print(f"  3. 02b_frequency_bands.png            - Detail coefficients by frequency")
print(f"  4. 03_anomaly_detection.png           - Anomaly analysis")
print(f"  5. 04_trading_signals.png             - Buy/sell signals")
print(f"  6. 05_statistics_dashboard.png        - Statistical summary")

print(f"\nAll plots saved to: {output_dir}/")
print(f"Processing time: {cuda_time*1000:.2f}ms (wavelet decomposition)")

if torch.cuda.is_available():
    print(f"\n✓ GPU Acceleration: CUDA on {torch.cuda.get_device_name(0)}")
else:
    print(f"\n⚠ GPU Acceleration: None (CPU mode)")

print("=" * 70)
