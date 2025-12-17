#!/usr/bin/env python3
"""Plot 1: Main Overview (4 subplots)"""
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from plot_common import load_or_compute_wavelet_data

CURRENCY = sys.argv[1].upper()
TIMEFRAME = sys.argv[2].lower()

def configure_date_axis(ax, tf_string):
    """Configure date axis formatting"""
    minutes_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
    minutes = minutes_map.get(tf_string, 5)
    
    if minutes <= 5:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    elif minutes < 1440:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Load data
data = load_or_compute_wavelet_data(CURRENCY, TIMEFRAME)
prices = data['prices']
dates = data['dates']
volumes = data['volumes']
trend_gpu = data['trend_gpu']
detail_gpu = data['detail_gpu']
anomaly_threshold = data['anomaly_threshold']
anomaly_indices = data['anomaly_indices']

# Generate plot
start_time = time.time()
output_dir = f'wavelet_plots/{CURRENCY.lower()}'

fig = plt.figure(figsize=(16, 12), dpi=300)
gs = GridSpec(4, 1, figure=fig, hspace=0.3)

# Interpolate trend back to original length
import numpy as np
trend_indices = np.linspace(0, len(prices)-1, len(trend_gpu))
original_indices = np.arange(len(prices))
interp_func = interp1d(trend_indices, trend_gpu, kind='cubic', fill_value='extrapolate')
trend_interpolated = interp_func(original_indices)

offset = len(prices) - len(trend_gpu)
aligned_dates = dates[offset:]

# Subplot 1: Original prices
ax1 = fig.add_subplot(gs[0])
ax1.plot(dates, prices, 'b-', linewidth=1, alpha=0.7, label='Original Price')
ax1.set_title(f'{CURRENCY}/USDT Price History ({TIMEFRAME} candles)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price (USD)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1.set_xlim(dates[0], dates[-1])
configure_date_axis(ax1, TIMEFRAME)

# Subplot 2: Price with Trend
ax2 = fig.add_subplot(gs[1])
ax2.plot(dates, prices, 'b-', linewidth=1, alpha=0.4, label='Original Price')
ax2.plot(dates, trend_interpolated, 'r-', linewidth=2, label='Trend')
ax2.set_title('Price Decomposition: Original vs Trend', fontsize=14, fontweight='bold')
ax2.set_ylabel('Price (USD)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')
ax2.set_xlim(dates[0], dates[-1])
configure_date_axis(ax2, TIMEFRAME)

# Subplot 3: Detail coefficients
ax3 = fig.add_subplot(gs[2])
ax3.plot(aligned_dates, detail_gpu, 'g-', linewidth=1, alpha=0.7, label='Detail')
ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax3.axhline(y=anomaly_threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold (${anomaly_threshold:.0f})')
ax3.axhline(y=-anomaly_threshold, color='r', linestyle='--', linewidth=1)
if len(anomaly_indices) > 0:
    anomaly_dates = [aligned_dates[i] for i in anomaly_indices if i < len(aligned_dates)]
    anomaly_values = detail_gpu[anomaly_indices[:len(anomaly_dates)]]
    ax3.scatter(anomaly_dates, anomaly_values, color='red', s=50, marker='*', zorder=5, label=f'Anomalies ({len(anomaly_indices)})')
ax3.set_title('Detail Coefficients', fontsize=14, fontweight='bold')
ax3.set_ylabel('Detail (USD)', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left')
ax3.set_xlim(aligned_dates[0], aligned_dates[-1])
configure_date_axis(ax3, TIMEFRAME)

# Subplot 4: Volume
ax4 = fig.add_subplot(gs[3])
ax4.bar(dates, volumes, width=0.04, color='purple', alpha=0.6)
ax4.set_title('Trading Volume', fontsize=14, fontweight='bold')
ax4.set_ylabel('Volume', fontsize=11)
ax4.set_xlabel('Date', fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(dates[0], dates[-1])
configure_date_axis(ax4, TIMEFRAME)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_main_overview.png', dpi=300, bbox_inches='tight')
plt.close()

elapsed = time.time() - start_time
print(f"[PLOT 1] Generated in {elapsed:.2f}s")
