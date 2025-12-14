"""
Test GPU wavelet implementation against PyWavelets (pywt) reference
"""

import numpy as np
import pywt
import pyopencl as cl
import ccxt
from datetime import datetime

# =============================================================================
# GPU SETUP
# =============================================================================

platforms = cl.get_platforms()
platform = [p for p in platforms if 'nvidia' in p.name.lower() or 'cuda' in p.name.lower()][0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

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
# FETCH TEST DATA
# =============================================================================

print("Fetching SOL/USDT data from Binance...")
exchange = ccxt.binance({'enableRateLimit': True})
ohlcv = exchange.fetch_ohlcv('SOL/USDT', '15m', limit=1000)
prices = np.array([candle[4] for candle in ohlcv], dtype=np.float32)

# Scale to [-1, 1] like in wave_nada.ipynb
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten().astype(np.float32)

print(f"Loaded {len(prices_scaled)} price points")
print(f"Range: {prices_scaled.min():.4f} to {prices_scaled.max():.4f}")

# =============================================================================
# PYWAVELETS REFERENCE (DB6, 9 LEVELS)
# =============================================================================

print("\n" + "=" * 70)
print("PYWAVELETS REFERENCE (DB6, 9 LEVELS)")
print("=" * 70)

wavelet = 'db6'
levels = 9
coeffs_pywt = pywt.wavedec(prices_scaled, wavelet, level=levels)

print(f"\nPyWavelets decomposition:")
print(f"  Approximation (cA{levels}): {len(coeffs_pywt[0])} points")
for i in range(1, len(coeffs_pywt)):
    print(f"  Detail cD{levels-i+1}: {len(coeffs_pywt[i])} points, "
          f"std={np.std(coeffs_pywt[i]):.6f}, "
          f"zero-crossings={np.sum(np.diff(np.sign(coeffs_pywt[i])) != 0)}")

# =============================================================================
# GPU IMPLEMENTATION TEST
# =============================================================================

print("\n" + "=" * 70)
print("GPU WAVELET DECOMPOSITION (MATCHING PYWT)")
print("=" * 70)

# Get DB6 wavelet filters from pywt
wavelet_obj = pywt.Wavelet('db6')
dec_lo = np.array(wavelet_obj.dec_lo, dtype=np.float32)  # Low-pass (approximation)
dec_hi = np.array(wavelet_obj.dec_hi, dtype=np.float32)  # High-pass (detail)

print(f"\nDB6 Wavelet coefficients:")
print(f"  Low-pass filter length: {len(dec_lo)}")
print(f"  High-pass filter length: {len(dec_hi)}")
print(f"  Low-pass sum: {dec_lo.sum():.6f}")
print(f"  High-pass sum: {dec_hi.sum():.6f}")

# Perform multi-level decomposition on GPU
print(f"\nGPU decomposition:")
current_signal = prices_scaled
gpu_approximations = []
gpu_details = []

for level in range(levels):
    # Check if signal is long enough for filter
    if len(current_signal) < len(dec_lo):
        print(f"  Level {level+1}: Signal too short ({len(current_signal)} < {len(dec_lo)}), stopping")
        break
    
    # Apply filters
    approx = gpu_convolve(current_signal, dec_lo, convolve_kernel)
    detail = gpu_convolve(current_signal, dec_hi, convolve_kernel)
    
    # PyWavelets downsamples by 2 after convolution (dyadic decomposition)
    approx = np.ascontiguousarray(approx[::2])  # Keep every other point
    detail = np.ascontiguousarray(detail[::2])  # Keep every other point
    
    gpu_approximations.append(approx)
    gpu_details.append(detail)
    
    print(f"  Level {level+1}:")
    print(f"    Approximation: {len(approx)} points, std={approx.std():.6f}")
    print(f"    Detail: {len(detail)} points, "
          f"std={detail.std():.6f}, "
          f"zero-crossings={np.sum(np.diff(np.sign(detail)) != 0)}")
    
    # Next level works on approximation
    current_signal = approx

actual_levels = len(gpu_details)

# =============================================================================
# COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("COMPARISON: GPU vs PyWavelets")
print("=" * 70)

# Compare final approximation
final_approx_pywt = coeffs_pywt[0]
final_approx_gpu = gpu_approximations[-1]

print(f"\nFinal Approximation (Level {levels}):")
print(f"  PyWavelets length: {len(final_approx_pywt)}")
print(f"  GPU length: {len(final_approx_gpu)}")
print(f"  Length match: {'✓' if len(final_approx_pywt) == len(final_approx_gpu) else '✗'}")

if len(final_approx_pywt) == len(final_approx_gpu):
    diff = np.abs(final_approx_pywt - final_approx_gpu)
    print(f"  Max difference: {diff.max():.10f}")
    print(f"  Mean difference: {diff.mean():.10f}")
    match = np.allclose(final_approx_pywt, final_approx_gpu, rtol=1e-5, atol=1e-8)
    print(f"  Values match: {'✓' if match else '✗'}")

# Compare detail coefficients
print(f"\nDetail Coefficients Comparison:")
print(f"  GPU completed {actual_levels} levels, PyWavelets has {levels} levels")
for level in range(min(actual_levels, levels)):
    detail_pywt = coeffs_pywt[level + 1]  # pywt stores [cA, cD_n, cD_n-1, ..., cD_1]
    detail_gpu = gpu_details[actual_levels - level - 1]  # GPU stores [cD_1, cD_2, ..., cD_n]
    
    print(f"\n  Level {level+1} (cD{level+1}):")
    print(f"    PyWavelets length: {len(detail_pywt)}")
    print(f"    GPU length: {len(detail_gpu)}")
    print(f"    Length match: {'✓' if len(detail_pywt) == len(detail_gpu) else '✗'}")
    
    if len(detail_pywt) == len(detail_gpu):
        diff = np.abs(detail_pywt - detail_gpu)
        print(f"    Max difference: {diff.max():.10f}")
        print(f"    Mean difference: {diff.mean():.10f}")
        match = np.allclose(detail_pywt, detail_gpu, rtol=1e-4, atol=1e-6)
        print(f"    Values match: {'✓' if match else '✗'}")
        
        # Compare statistics
        print(f"    PyWavelets std: {detail_pywt.std():.6f}")
        print(f"    GPU std: {detail_gpu.std():.6f}")
        print(f"    PyWavelets zero-crossings: {np.sum(np.diff(np.sign(detail_pywt)) != 0)}")
        print(f"    GPU zero-crossings: {np.sum(np.diff(np.sign(detail_gpu)) != 0)}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
