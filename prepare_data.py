#!/usr/bin/env python3
"""
Data preparation script - loads data, performs GPU wavelet decomposition, saves results.
Run this first, then plot scripts can load cached data and generate plots in parallel.
"""
import pyopencl as cl
import numpy as np
import ccxt
from datetime import datetime
import sys
import time
from plot_common import gpu_convolve, save_wavelet_data

# Parse arguments
if len(sys.argv) < 3:
    print("Usage: prepare_data.py <CURRENCY> <TIMEFRAME>")
    sys.exit(1)

CURRENCY = sys.argv[1].upper()
TIMEFRAME = sys.argv[2].lower()

print(f"[DATA PREP] Preparing wavelet data for {CURRENCY}/{TIMEFRAME}")

# Initialize OpenCL
platforms = cl.get_platforms()
platform = platforms[0]
devices = platform.get_devices()
device = devices[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

# Build convolution kernel
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

# Define wavelets
haar_low_pass = np.array([0.5, 0.5], dtype=np.float32)
haar_high_pass = np.array([0.5, -0.5], dtype=np.float32)

# Fetch data from Binance
data_load_start = time.time()
exchange = ccxt.binance({'enableRateLimit': True})
symbol = f'{CURRENCY}/USDT'

# Fetch recent data (last 7 days)
from datetime import timedelta
since_date = datetime.now() - timedelta(days=7)
since = exchange.parse8601(since_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'))
all_data = []

while True:
    data = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
    if len(data) == 0:
        break
    all_data.extend(data)
    since = data[-1][0] + 1
    if len(data) < 1000:
        break

timestamps = [d[0] for d in all_data]
prices = np.array([d[4] for d in all_data], dtype=np.float32)  # Close prices
volumes = np.array([d[5] for d in all_data], dtype=np.float32)
dates = [datetime.fromtimestamp(t / 1000) for t in timestamps]

data_load_time = time.time() - data_load_start
print(f"[DATA PREP] Loaded {len(prices)} data points in {data_load_time:.2f}s")

# Perform wavelet decomposition
wavelet_start = time.time()

trend_gpu = gpu_convolve(prices, haar_low_pass, convolve_kernel, ctx, queue, mode='symmetric')
detail_gpu = gpu_convolve(prices, haar_high_pass, convolve_kernel, ctx, queue, mode='symmetric')

# Multi-level decomposition
approximations = []
details = []
current_signal = prices

for i in range(8):
    if len(current_signal) < len(haar_low_pass):
        break
    
    approx = gpu_convolve(current_signal, haar_low_pass, convolve_kernel, ctx, queue, mode='symmetric')
    detail = gpu_convolve(current_signal, haar_high_pass, convolve_kernel, ctx, queue, mode='symmetric')
    
    approximations.append(approx)
    details.append(detail)
    current_signal = approx

wavelet_time = time.time() - wavelet_start

# Anomaly detection
detail_abs = np.abs(detail_gpu)
median = np.median(detail_abs)
mad = np.median(np.abs(detail_abs - median))
threshold = 6.0
anomaly_threshold = median + threshold * mad
anomaly_indices = np.where(detail_abs > anomaly_threshold)[0]

# Prepare data package
data_dict = {
    'prices': prices,
    'dates': dates,
    'volumes': volumes,
    'trend_gpu': trend_gpu,
    'detail_gpu': detail_gpu,
    'detail_abs': detail_abs,
    'approximations': approximations,
    'details': details,
    'anomaly_threshold': anomaly_threshold,
    'anomaly_indices': anomaly_indices,
    'median': median,
    'CURRENCY': CURRENCY,
    'TIMEFRAME': TIMEFRAME,
    'device_name': device.name,
    'data_load_time': data_load_time,
    'wavelet_time': wavelet_time
}

# Save to disk
save_wavelet_data(CURRENCY, data_dict)

print(f"[DATA PREP] Wavelet computation: {wavelet_time:.2f}s")
print(f"[DATA PREP] Data saved successfully")
print(f"[DATA PREP] Total: {data_load_time + wavelet_time:.2f}s")
