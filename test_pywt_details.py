"""
Investigate PyWavelets internals to understand exact behavior
"""

import numpy as np
import pywt

# Simple test signal
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

print("=" * 70)
print("PYWAVELETS INTERNALS INVESTIGATION")
print("=" * 70)

# Test with db6
wavelet = pywt.Wavelet('db6')
print(f"\nDB6 Wavelet:")
print(f"  Dec lo (low-pass): {wavelet.dec_lo}")
print(f"  Dec hi (high-pass): {wavelet.dec_hi}")
print(f"  Filter length: {wavelet.dec_len}")

# Single level decomposition
print(f"\n\nSingle Level Decomposition:")
print(f"Input signal: {signal}")
print(f"Input length: {len(signal)}")

# Default mode
cA, cD = pywt.dwt(signal, 'db6')
print(f"\nDefault mode:")
print(f"  cA length: {len(cA)}, values: {cA}")
print(f"  cD length: {len(cD)}, values: {cD}")

# Try different modes
for mode in ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']:
    try:
        cA, cD = pywt.dwt(signal, 'db6', mode=mode)
        print(f"\nMode '{mode}':")
        print(f"  cA length: {len(cA)}")
        print(f"  cD length: {len(cD)}")
    except Exception as e:
        print(f"\nMode '{mode}': Error - {e}")

# Multi-level with different modes
print(f"\n\n{'=' * 70}")
print("MULTI-LEVEL DECOMPOSITION MODES")
print("=" * 70)

test_signal = np.random.randn(100)
for mode in ['zero', 'symmetric', 'periodization']:
    coeffs = pywt.wavedec(test_signal, 'db6', level=3, mode=mode)
    print(f"\nMode '{mode}' (3 levels, 100 points):")
    print(f"  cA3: {len(coeffs[0])} points")
    print(f"  cD3: {len(coeffs[1])} points")
    print(f"  cD2: {len(coeffs[2])} points")
    print(f"  cD1: {len(coeffs[3])} points")

# Check what wavedec actually does
print(f"\n\n{'=' * 70}")
print("EXAMINING WAVEDEC DEFAULT BEHAVIOR")
print("=" * 70)

# Use 1000 points like in our test
np.random.seed(42)
test_signal = np.random.randn(1000)
coeffs = pywt.wavedec(test_signal, 'db6', level=9)

print(f"\nInput: 1000 points")
print(f"Output:")
print(f"  cA9: {len(coeffs[0])} points")
for i in range(1, len(coeffs)):
    print(f"  cD{10-i}: {len(coeffs[i])} points")

# Calculate expected lengths with downsampling
print(f"\n\nExpected lengths with convolution + downsample by 2:")
current_len = 1000
filter_len = 12
for level in range(1, 10):
    after_conv = current_len - filter_len + 1
    after_downsample = after_conv // 2
    print(f"  Level {level}: {current_len} → (conv) → {after_conv} → (↓2) → {after_downsample}")
    current_len = after_downsample
    if current_len < filter_len:
        print(f"  Level {level+1}: Too short!")
        break
