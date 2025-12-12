#!/usr/bin/env python3
"""
Minimal import test to identify what's causing bus errors on Pi Zero
"""
import sys

print("Testing imports one by one...")
print()

try:
    print("[1/6] Testing sys, os, time... ", end='', flush=True)
    import os
    import time
    from datetime import datetime
    print("✓")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

try:
    print("[2/6] Testing numpy... ", end='', flush=True)
    import numpy as np
    print("✓")
    
    print("      - Creating array... ", end='', flush=True)
    test_arr = np.array([1, 2, 3, 4, 5])
    print("✓")
    
    print("      - Testing operations... ", end='', flush=True)
    result = np.convolve(test_arr, [0.5, 0.5], mode='valid')
    print("✓")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

try:
    print("[3/6] Testing ccxt... ", end='', flush=True)
    import ccxt
    print("✓")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

try:
    print("[4/6] Testing matplotlib backend... ", end='', flush=True)
    import matplotlib
    matplotlib.use('Agg')
    print("✓")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

try:
    print("[5/6] Testing matplotlib.pyplot... ", end='', flush=True)
    import matplotlib.pyplot as plt
    print("✓")
except Exception as e:
    print(f"✗ FAILED: {e}")
    print("\n⚠ This is likely where the bus error occurs on Pi Zero")
    sys.exit(1)

try:
    print("[6/6] Testing plot creation... ", end='', flush=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot([1, 2, 3], [1, 4, 9])
    plt.close()
    print("✓")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

print()
print("=" * 50)
print("✓ ALL IMPORTS SUCCESSFUL")
print("=" * 50)
