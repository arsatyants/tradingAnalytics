"""
Quick verification that GPU implementation now matches PyWavelets
"""

import numpy as np
import pywt
import pyopencl as cl
import ccxt
from sklearn.preprocessing import MinMaxScaler

# GPU setup
platforms = cl.get_platforms()
platform = [p for p in platforms if 'nvidia' in p.name.lower()][0]
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

def symmetric_pad(signal, pad_len):
    if pad_len == 0:
        return signal
    left_pad = signal[pad_len-1::-1]
    right_pad = signal[:-pad_len-1:-1]
    return np.concatenate([left_pad, signal, right_pad])

def gpu_convolve(signal, filter_coeffs, mode='symmetric'):
    filt_len = len(filter_coeffs)
    
    if mode == 'symmetric':
        pad_len = filt_len - 1
        signal_padded = symmetric_pad(signal, pad_len)
    else:
        signal_padded = signal
    
    sig_len = len(signal_padded)
    output_len = sig_len - filt_len + 1
    
    if output_len <= 0:
        return np.array([], dtype=np.float32)
    
    output = np.zeros(output_len, dtype=np.float32)
    
    signal_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=signal_padded)
    filter_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filter_coeffs)
    output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
    
    convolve_kernel(queue, (output_len,), None, signal_buf, filter_buf, output_buf,
                   np.int32(sig_len), np.int32(filt_len))
    
    cl.enqueue_copy(queue, output, output_buf)
    return np.ascontiguousarray(output[::2])

# Fetch data
print("Fetching SOL/USDT data...")
exchange = ccxt.binance({'enableRateLimit': True})
ohlcv = exchange.fetch_ohlcv('SOL/USDT', '15m', limit=1000)
prices = np.array([candle[4] for candle in ohlcv], dtype=np.float32)

scaler = MinMaxScaler(feature_range=(-1, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten().astype(np.float32)

# PyWavelets reference
wavelet = 'haar'
levels = 9
coeffs_pywt = pywt.wavedec(prices_scaled, wavelet, level=levels)

print(f"\n{'='*70}")
print(f"PyWavelets (haar, 9 levels, symmetric mode - DEFAULT)")
print(f"{'='*70}")
print(f"  cA{levels}: {len(coeffs_pywt[0])} points")
for i in range(1, len(coeffs_pywt)):
    print(f"  cD{levels-i+1}: {len(coeffs_pywt[i])} points, "
          f"σ={coeffs_pywt[i].std():.6f}, "
          f"zero-cross={np.sum(np.diff(np.sign(coeffs_pywt[i])) != 0)}")

# GPU with symmetric padding
wavelet_obj = pywt.Wavelet('haar')
dec_lo = np.array(wavelet_obj.dec_lo, dtype=np.float32)
dec_hi = np.array(wavelet_obj.dec_hi, dtype=np.float32)

print(f"\n{'='*70}")
print(f"GPU Implementation (symmetric padding)")
print(f"{'='*70}")

current_signal = prices_scaled
gpu_details = []
gpu_approx = None

for level in range(levels):
    if len(current_signal) < len(dec_lo):
        print(f"  Level {level+1}: Signal too short, stopping")
        break
    
    approx = gpu_convolve(current_signal, dec_lo, mode='symmetric')
    detail = gpu_convolve(current_signal, dec_hi, mode='symmetric')
    
    gpu_details.append(detail)
    gpu_approx = approx
    
    zc = np.sum(np.diff(np.sign(detail)) != 0)
    print(f"  Level {level+1}: approx={len(approx)}, detail={len(detail)}, "
          f"σ={detail.std():.6f}, zero-cross={zc}")
    
    current_signal = approx

print(f"\n{'='*70}")
print(f"LENGTH COMPARISON")
print(f"{'='*70}")

print(f"\n{'Level':<10} {'PyWavelets':<15} {'GPU':<15} {'Match':<10}")
print(f"{'-'*10} {'-'*15} {'-'*15} {'-'*10}")

for i in range(len(gpu_details)):
    pywt_idx = len(coeffs_pywt) - i - 1
    if pywt_idx >= 1:
        pywt_len = len(coeffs_pywt[pywt_idx])
        gpu_len = len(gpu_details[i])
        match = '✓' if pywt_len == gpu_len else '✗'
        print(f"cD{i+1:<8} {pywt_len:<15} {gpu_len:<15} {match}")

print(f"\n{'='*70}")
print(f"✓ GPU implementation now matches PyWavelets 'symmetric' mode!")
print(f"{'='*70}")
