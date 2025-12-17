"""
Shared utilities and data loading for parallel plot generation
"""
import pyopencl as cl
import numpy as np
import ccxt
from datetime import datetime
import os
import sys
import pickle

def symmetric_pad(signal, pad_len):
    """Apply symmetric padding to match PyWavelets 'symmetric' mode."""
    if pad_len == 0:
        return signal
    left_pad = signal[pad_len-1::-1]
    right_pad = signal[:-pad_len-1:-1]
    return np.concatenate([left_pad, signal, right_pad])

def gpu_convolve(signal, filter_coeffs, kernel, ctx, queue, mode='symmetric'):
    """Perform convolution on GPU with boundary handling."""
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
    
    kernel(queue, (output_len,), None, signal_buf, filter_buf, output_buf,
          np.int32(sig_len), np.int32(filt_len))
    
    cl.enqueue_copy(queue, output, output_buf)
    return np.ascontiguousarray(output[::2])

def load_or_compute_wavelet_data(currency, timeframe):
    """Load pre-computed wavelet data or compute if missing."""
    cache_file = f'wavelet_plots/{currency.lower()}/wavelet_data.pkl'
    
    if os.path.exists(cache_file):
        # Load cached data
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Data not cached - need to compute
    # This should only happen if data preparation script wasn't run
    raise FileNotFoundError(f"Wavelet data not found: {cache_file}. Run data preparation first.")

def save_wavelet_data(currency, data_dict):
    """Save computed wavelet data for reuse by plot scripts."""
    output_dir = f'wavelet_plots/{currency.lower()}'
    os.makedirs(output_dir, exist_ok=True)
    cache_file = f'{output_dir}/wavelet_data.pkl'
    
    with open(cache_file, 'wb') as f:
        pickle.dump(data_dict, f)
