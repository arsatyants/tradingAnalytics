"""
GPU-Accelerated Wavelet Decomposition with High-Quality Graphics
================================================================

This script generates publication-quality PNG plots of wavelet analysis
using matplotlib instead of ASCII console graphics.

Features:
- High-resolution plots (300 DPI)
- Multiple subplots for comprehensive analysis
- Real BTC data from Binance
- GPU-accelerated wavelet computations
- Saves all plots to 'wavelet_plots/' directory
"""

import pyopencl as cl
import numpy as np
import time
import ccxt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import os

# Create output directory
output_dir = 'wavelet_plots'
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("GPU-ACCELERATED WAVELET DECOMPOSITION - GRAPHICS MODE")
print("=" * 70)

# =============================================================================
# STEP 1: INITIALIZE OPENCL WITH AUTO-DETECTION
# =============================================================================

def detect_best_opencl_platform():
    """
    Auto-detect the best OpenCL platform available.
    Priority: NVIDIA CUDA > AMD > Mesa Rusticl (ARM Mali) > Other
    """
    platforms = cl.get_platforms()
    
    if not platforms:
        raise RuntimeError("No OpenCL platforms found!")
    
    print(f"\n🔍 Detected {len(platforms)} OpenCL platform(s):")
    for i, p in enumerate(platforms):
        devices = p.get_devices()
        print(f"   [{i}] {p.name} - {len(devices)} device(s)")
        for d in devices:
            print(f"       └─ {d.name} ({d.max_compute_units} CUs)")
    
    # Priority 1: NVIDIA CUDA
    for p in platforms:
        if 'nvidia' in p.name.lower() or 'cuda' in p.name.lower():
            print(f"\n✓ Selected: {p.name} (NVIDIA GPU detected)")
            return p
    
    # Priority 2: AMD
    for p in platforms:
        if 'amd' in p.name.lower() or 'advanced micro devices' in p.name.lower():
            print(f"\n✓ Selected: {p.name} (AMD GPU detected)")
            return p
    
    # Priority 3: Mesa Rusticl (ARM Mali via Panfrost)
    for p in platforms:
        if 'rusticl' in p.name.lower() or 'mesa' in p.name.lower():
            print(f"\n✓ Selected: {p.name} (ARM Mali GPU detected)")
            return p
    
    # Priority 4: Intel
    for p in platforms:
        if 'intel' in p.name.lower():
            print(f"\n✓ Selected: {p.name} (Intel GPU detected)")
            return p
    
    # Fallback: Use first available platform
    print(f"\n✓ Selected: {platforms[0].name} (default)")
    return platforms[0]

platform = detect_best_opencl_platform()
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

print(f"  Device: {device.name}")
print(f"  Compute Units: {device.max_compute_units}")
print(f"  Max Work Group Size: {device.max_work_group_size}")
print(f"  Global Memory: {device.global_mem_size / 1024**3:.2f} GB")
print(f"  Output Directory: {output_dir}/\n")

# =============================================================================
# STEP 2: DEFINE WAVELET KERNEL
# =============================================================================

convolution_kernel = """
__kernel void convolve(__global const float *signal,
                       __global const float *filter,
                       __global float *output,
                       const int sig_len,
                       const int filt_len) {
    int i = get_global_id(0);
    if(i >= sig_len - filt_len + 1) return;
    
    float sum = 0.0f;
    for(int j = 0; j < filt_len; j++) {
        sum += signal[i + j] * filter[j];
    }
    output[i] = sum;
}
"""

program = cl.Program(ctx, convolution_kernel).build()
convolve_kernel = cl.Kernel(program, "convolve")

# =============================================================================
# STEP 3: DEFINE WAVELETS
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
# STEP 4: GPU CONVOLUTION FUNCTION
# =============================================================================

def gpu_convolve(signal, filter_coeffs, kernel):
    sig_len = len(signal)
    filt_len = len(filter_coeffs)
    output_len = sig_len - filt_len + 1
    output = np.zeros(output_len, dtype=np.float32)
    
    signal_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=signal)
    filter_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filter_coeffs)
    output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
    
    kernel(queue, (output_len,), None, signal_buf, filter_buf, output_buf,
          np.int32(sig_len), np.int32(filt_len))
    
    cl.enqueue_copy(queue, output, output_buf)
    return output

# =============================================================================
# STEP 5: FETCH REAL BTC DATA
# =============================================================================

print("=" * 70)
print("FETCHING BTC DATA FROM BINANCE")
print("=" * 70)

try:
    from datetime import timedelta
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
    prices = np.array([candle[4] for candle in ohlcv], dtype=np.float32)
    volumes = np.array([candle[5] for candle in ohlcv], dtype=np.float32)
    dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]
    
    print(f"  Data range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Current BTC: ${prices[-1]:,.2f}")
    print(f"  Change: {((prices[-1] - prices[0]) / prices[0] * 100):+.2f}%\n")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    exit(1)

# =============================================================================
# STEP 6: PERFORM WAVELET DECOMPOSITION
# =============================================================================

print("=" * 70)
print("COMPUTING WAVELET DECOMPOSITION")
print("=" * 70)

start = time.time()
trend_gpu = gpu_convolve(prices, haar_low_pass, convolve_kernel)
detail_gpu = gpu_convolve(prices, haar_high_pass, convolve_kernel)
trend_db4 = gpu_convolve(prices, db4_low_pass, convolve_kernel)
gpu_time = time.time() - start

print(f"\n✓ Decomposition complete: {gpu_time*1000:.2f}ms")
print(f"  Trend points: {len(trend_gpu)}")
print(f"  Detail points: {len(detail_gpu)}\n")

# Multi-level decomposition (proper wavelet pyramid)
# Each level shows: approximation (trend) and detail at that scale
approximations = []  # Low-pass (smoothed trend)
details = []         # High-pass (detail at each scale)

current_signal = prices

for i in range(5):
    # Apply both low-pass and high-pass filters
    approx = gpu_convolve(current_signal, haar_low_pass, convolve_kernel)
    detail = gpu_convolve(current_signal, haar_high_pass, convolve_kernel)
    
    approximations.append(approx)
    details.append(detail)
    
    # Next level operates on the approximation (coarser scale)
    current_signal = approx

# For compatibility, keep 'levels' as approximations
levels = approximations

# =============================================================================
# STEP 7: ANOMALY DETECTION
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

print("\n[1/5] Main overview plot... ", end='', flush=True)

fig = plt.figure(figsize=(16, 12), dpi=300)
gs = GridSpec(4, 1, figure=fig, hspace=0.3)

# Align dates
offset = len(prices) - len(trend_gpu)
aligned_dates = dates[offset:]

# Subplot 1: Original prices
ax1 = fig.add_subplot(gs[0])
ax1.plot(dates, prices, 'b-', linewidth=1, alpha=0.7, label='Original Price')
ax1.set_title('BTC/USDT Price History (1-hour candles)', fontsize=14, fontweight='bold')
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
# Mark anomalies
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

# =============================================================================
# PLOT 2A: PROGRESSIVE APPROXIMATIONS
# =============================================================================

print("[2/6] Progressive approximations... ", end='', flush=True)

fig = plt.figure(figsize=(24, 20), dpi=300)
gs = GridSpec(6, 1, figure=fig, hspace=0.35)
fig.suptitle('Progressive Approximations - Frequency Filtering', fontsize=20, fontweight='bold')

# Original signal at top
ax_orig = plt.subplot(gs[0])
ax_orig.plot(dates, prices, 'b-', linewidth=1.5, alpha=0.8, label='Original (all frequencies)')
ax_orig.set_title('Original Signal: BTC/USDT', fontsize=13, fontweight='bold')
ax_orig.set_ylabel('Price (USD)', fontsize=11)
ax_orig.grid(True, alpha=0.3)
ax_orig.legend(loc='upper left')
ax_orig.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Frequency bands explanation
freq_bands = [
    ('1-2 hours', '~50% of frequencies', 'Noise & very short-term fluctuations'),
    ('2-4 hours', '~25% of frequencies', 'Short-term trading movements'),
    ('4-8 hours', '~12.5% of frequencies', 'Intraday trend changes'),
    ('8-16 hours', '~6.25% of frequencies', 'Daily cycle patterns'),
    ('16+ hours (days)', '~3.125% of frequencies', 'Multi-day trend reversals')
]

# Plot progressive approximations with overlays showing differences
approx_colors = ['orangered', 'orange', 'gold', 'limegreen', 'dodgerblue']
approx_labels = [
    'After removing 1-2h noise',
    'After removing up to 4h fluctuations',
    'After removing up to 8h movements',
    'After removing up to 16h variations',
    'Main trend only (16h+ timescale)'
]

for i in range(5):
    ax_approx = plt.subplot(gs[i+1])
    
    offset_a = len(prices) - len(approximations[i])
    dates_a = dates[offset_a:]
    current_approx = approximations[i]
    
    # Plot previous level (if exists) to show what's being removed
    if i > 0:
        offset_prev = len(prices) - len(approximations[i-1])
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
        original_segment = prices[offset_a:offset_a+len(approximations[i])]
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
        original_segment = prices[offset_a:offset_a+len(approximations[i])]
        smoothness_increase = original_segment.std() / (current_approx.std() + 1e-10)
        removed_variance = difference.std()
    
    ax_approx.set_title(f'Level {i+1}: {approx_labels[i]}', 
                       fontsize=12, fontweight='bold', pad=10)
    ax_approx.set_ylabel('Price (USD)', fontsize=10)
    ax_approx.grid(True, alpha=0.3)
    ax_approx.legend(loc='upper left', fontsize=8)
    ax_approx.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
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
ax_orig.plot(dates, prices, 'b-', linewidth=1.5, alpha=0.8)
ax_orig.set_title('Original Signal: BTC/USDT (All Frequencies Combined)', fontsize=13, fontweight='bold')
ax_orig.set_ylabel('Price (USD)', fontsize=11)
ax_orig.grid(True, alpha=0.3)
ax_orig.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
price_range = prices.max() - prices.min()
ax_orig.text(0.02, 0.95, f'Range: ${price_range:,.0f}\nStd: ${prices.std():,.0f}', 
            transform=ax_orig.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot each detail band
colors_map = ['red', 'orange', 'gold', 'green', 'blue']

for i in range(5):
    ax_detail = plt.subplot(gs[i+1])
    
    offset_d = len(prices) - len(details[i])
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
    ax_detail.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
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
# PLOT 3: ANOMALY DETECTION DETAIL
# =============================================================================

print("[4/6] Anomaly detection detail... ", end='', flush=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=300)
fig.suptitle('Anomaly Detection Analysis', fontsize=16, fontweight='bold')

# Detail coefficients with anomalies
ax1.plot(aligned_dates, detail_gpu, 'b-', linewidth=1, alpha=0.7, label='Detail Coefficients')
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax1.axhline(y=anomaly_threshold, color='r', linestyle='--', linewidth=1.5, 
           label=f'Threshold: ${anomaly_threshold:.2f}')
ax1.axhline(y=-anomaly_threshold, color='r', linestyle='--', linewidth=1.5)
ax1.fill_between(aligned_dates, anomaly_threshold, detail_gpu.max(), alpha=0.1, color='red')
ax1.fill_between(aligned_dates, -anomaly_threshold, detail_gpu.min(), alpha=0.1, color='red')

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

# Absolute detail (volatility measure)
ax2.plot(aligned_dates, detail_abs, 'purple', linewidth=1, alpha=0.7, label='Absolute Detail')
ax2.axhline(y=anomaly_threshold, color='r', linestyle='--', linewidth=1.5, 
           label=f'Threshold: ${anomaly_threshold:.2f}')
ax2.fill_between(aligned_dates, 0, anomaly_threshold, alpha=0.2, color='green', label='Normal Range')
ax2.fill_between(aligned_dates, anomaly_threshold, detail_abs.max(), alpha=0.2, color='red', label='Anomaly Zone')

ax2.set_title('Volatility Measure (Absolute Detail Coefficients)', fontsize=13)
ax2.set_ylabel('Absolute Detail (USD)', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.subplots_adjust(hspace=0.3)
plt.savefig(f'{output_dir}/03_anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 4: TRADING SIGNALS
# =============================================================================

print("[5/6] Trading signals... ", end='', flush=True)

# Generate signals
trend = levels[2]
offset = len(prices) - len(trend)
aligned_prices = prices[offset:]
signal_dates = dates[offset:]
price_deviation = aligned_prices - trend
buffer = np.std(price_deviation) * 0.5

buy_signals = price_deviation < -buffer
sell_signals = price_deviation > buffer

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=300)
fig.suptitle('Trading Signal Generation', fontsize=16, fontweight='bold')

# Price with signals
ax1.plot(signal_dates, aligned_prices, 'b-', linewidth=1.5, alpha=0.7, label='Price')
ax1.plot(signal_dates, trend, 'orange', linewidth=2, label='Trend (Level 3)')
ax1.fill_between(signal_dates, trend - buffer, trend + buffer, alpha=0.2, color='gray', label='Neutral Zone')

# Mark buy/sell signals
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

# Deviation from trend
ax2.plot(signal_dates, price_deviation, 'purple', linewidth=1.5, alpha=0.7, label='Price Deviation')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.axhline(y=buffer, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Sell Threshold')
ax2.axhline(y=-buffer, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Buy Threshold')
ax2.fill_between(signal_dates, -buffer, buffer, alpha=0.2, color='gray')

ax2.scatter(buy_dates, price_deviation[buy_signals], color='green', s=80, marker='^', zorder=5)
ax2.scatter(sell_dates, price_deviation[sell_signals], color='red', s=80, marker='v', zorder=5)

ax2.set_title('Deviation from Trend', fontsize=13)
ax2.set_ylabel('Deviation (USD)', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.subplots_adjust(hspace=0.3)
plt.savefig(f'{output_dir}/04_trading_signals.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 5: STATISTICS DASHBOARD
# =============================================================================

print("[6/6] Statistics dashboard... ", end='', flush=True)

fig = plt.figure(figsize=(16, 10), dpi=300)
gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

# Histogram of detail coefficients
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(detail_gpu, bins=50, color='blue', alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax1.axvline(x=anomaly_threshold, color='red', linestyle=':', linewidth=2, label='Anomaly Threshold')
ax1.axvline(x=-anomaly_threshold, color='red', linestyle=':', linewidth=2)
ax1.set_title('Distribution of Detail Coefficients', fontsize=12, fontweight='bold')
ax1.set_xlabel('Detail Coefficient (USD)', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Price change distribution
ax2 = fig.add_subplot(gs[0, 1])
price_changes = np.diff(prices)
ax2.hist(price_changes, bins=50, color='green', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_title('Distribution of Hourly Price Changes', fontsize=12, fontweight='bold')
ax2.set_xlabel('Price Change (USD)', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.grid(True, alpha=0.3)

# Rolling volatility
ax3 = fig.add_subplot(gs[1, :])
window = 24  # 24-hour rolling window
rolling_std = np.array([detail_abs[max(0, i-window):i+1].std() for i in range(len(detail_abs))])
ax3.plot(aligned_dates, rolling_std, 'red', linewidth=2, label='24h Rolling Volatility')
ax3.fill_between(aligned_dates, 0, rolling_std, alpha=0.3, color='red')
ax3.set_title('Rolling Volatility (24-hour window)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Volatility (USD)', fontsize=10)
ax3.set_xlabel('Date', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Statistics table
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

stats_data = [
    ['Metric', 'Value'],
    ['', ''],
    ['Price Statistics', ''],
    ['  Current Price', f'${prices[-1]:,.2f}'],
    ['  Price Range', f'${prices.min():,.2f} - ${prices.max():,.2f}'],
    ['  Total Change', f'{((prices[-1] - prices[0]) / prices[0] * 100):+.2f}%'],
    ['  Std Deviation', f'${prices.std():,.2f}'],
    ['', ''],
    ['Detail Coefficients', ''],
    ['  Mean', f'${detail_gpu.mean():+.2f}'],
    ['  Std Deviation', f'${detail_gpu.std():.2f}'],
    ['  Median Abs Value', f'${median:.2f}'],
    ['  Max Abs Value', f'${detail_abs.max():.2f}'],
    ['', ''],
    ['Anomalies', ''],
    ['  Total Detected', f'{len(anomaly_indices)}'],
    ['  Anomaly Rate', f'{len(anomaly_indices)/len(detail_gpu)*100:.2f}%'],
    ['  Threshold', f'${anomaly_threshold:.2f}'],
    ['', ''],
    ['Trading Signals', ''],
    ['  Buy Signals', f'{buy_signals.sum()}'],
    ['  Sell Signals', f'{sell_signals.sum()}'],
    ['  Hold Periods', f'{len(trend) - buy_signals.sum() - sell_signals.sum()}'],
]

table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                 colWidths=[0.4, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style section headers
for row in [2, 8, 14, 18]:
    for col in range(2):
        table[(row, col)].set_facecolor('#E8F5E9')
        table[(row, col)].set_text_props(weight='bold')

plt.suptitle('Statistical Summary Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig(f'{output_dir}/05_statistics_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\n✓ All plots saved to '{output_dir}/' directory")
print(f"\nGenerated files:")
print(f"  1. 01_main_overview.png               - 4-panel overview")
print(f"  2. 02a_progressive_approximations.png - Progressive filtering (accumulations)")
print(f"  3. 02b_frequency_bands.png            - Detail coefficients by frequency")
print(f"  4. 03_anomaly_detection.png           - Anomaly analysis")
print(f"  5. 04_trading_signals.png             - Buy/sell signals")
print(f"  6. 05_statistics_dashboard.png        - Statistical summary")
print(f"\nGPU Processing Time: {gpu_time*1000:.2f}ms")
print(f"Device: {device.name}")
print("=" * 70)
