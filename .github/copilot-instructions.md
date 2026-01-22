# Trading Analytics - AI Agent Instructions

## Quick Reference

**Start Here:**
- Run analysis: `python web_server_parallel.py` → http://localhost:8080
- Single script: `python gpu_wavelet_gpu_plot.py BTC 5m`
- Setup: `./setup_env.sh` (creates .venv, installs deps, detects GPU)

**Key Constraints:**
- LSTM scaling: Always use `MinMaxScaler(feature_range=(-1, 1))` - never [0,1]
- Data loading: Adaptive date ranges by timeframe (see pattern below)
- OpenCL: Auto-detect best platform with scoring (NVIDIA>AMD>ARM)
- Pi Zero: Use `--demo --no-plots` flag (ccxt causes bus errors)

**File Organization:**
- Notebooks: Monolithic, state-dependent (must run sequentially)
- GPU scripts: Accept `<CURRENCY> <TIMEFRAME>` CLI args
- Parallel system: `prepare_data.py` → 6 plot processes (3.5x speedup)

## Project Overview
Cryptocurrency trading toolkit for **Orange Pi and embedded systems**. Performs BTC/ETH/SOL price prediction and anomaly detection using LSTM networks and GPU-accelerated wavelet transforms. Real-time data from Binance via CCXT.

**Key Design Goals:**
- Multi-platform GPU support (OpenCL → CUDA → Vulkan → OpenGL → CPU fallback)
- Graceful degradation (demo mode, lazy imports, error handling)
- Parallel processing (3.5x speedup via OS-level subprocess spawning)
- Publication-quality plot generation (300 DPI PNG, 6 plots per analysis)

## Architecture

### Component Hierarchy
```
Jupyter Notebooks (btc-prediction.ipynb, wave_nada.ipynb)
    ↓ LSTM training, interactive analysis
    
GPU Wavelet Scripts (5 variants: opencl/cuda/vulkan/opengl/cpu)
    ↓ Batch processing with CLI args: python <script>.py <CURRENCY> <TIMEFRAME>
    
Web Servers (3 variants: standard/parallel/vulkan)
    ↓ HTTP interface (port 8080) with async job tracking
    
Parallel Plot System (PRODUCTION)
    prepare_data.py → [plot_01.py, plot_template.py × 5] (subprocess.Popen)
    ↓ 3.5x speedup via OS-level parallelism (13s vs 45s)
```

### Core Files

**Notebooks** (monolithic, state-dependent):
- `btc-prediction.ipynb`: LSTM + CWT spectrograms + anomaly detection
- `wave_nada.ipynb`: DWT + Stochastic Oscillator + Nadaraya-Watson smoothing

**GPU Scripts** (accept `<CURRENCY> <TIMEFRAME>` args):
- `gpu_wavelet_gpu_plot.py`: OpenCL with auto-detection (NVIDIA→AMD→ARM Mali)
- `gpu_wavelet_cuda_plot.py`: PyTorch/CUDA with CPU fallback
- `gpu_wavelet_vulkan.py`: Vulkan compute shaders (most portable)
- `gpu_wavelet_opengl_plot.py`: OpenGL compute (Raspberry Pi 5 VideoCore VII)
- `gpu_wavelet_cpu_plot.py`: Pure NumPy, `--demo --no-plots` for Pi Zero

**Web Servers** (port 8080):
- `web_server.py`: Standard (sequential subprocess)
- `web_server_parallel.py`: **PREFERRED** - parallel plot generation (3.5x faster)
- `web_server_vulkan.py`: Vulkan backend variant

**Parallel System** (PRODUCTION):
- `prepare_data.py`: Data load + GPU wavelet → `wavelet_data.pkl` cache
- `plot_common.py`: Shared utilities (`gpu_convolve`, `load_wavelet_data`, `save_wavelet_data`)
- `plot_01.py`: Main 4-panel overview (full implementation)
- `plot_template.py`: Simplified plots 02a-05 (accepts plot ID arg)

### Execution Patterns
```bash
# Single currency analysis
python gpu_wavelet_gpu_plot.py BTC 5m    # Generates 6 PNG plots

# Batch processing (all currencies)
./run_all_currencies.sh                  # 18 plots (BTC/ETH/SOL × 6)
./run_all_currencies_console.sh          # Console output with ASCII

# Web interface (RECOMMENDED)
python web_server_parallel.py            # http://localhost:8080 (parallel)
python web_server.py                     # http://localhost:8080 (sequential)

# Parallel manual execution
python prepare_data.py BTC 5m && \
  python plot_01.py BTC 5m & \
  python plot_template.py BTC 5m 02a & wait

# Pi Zero safe mode (ccxt causes bus errors)
python gpu_wavelet_cpu_plot.py --demo --no-plots
```

## Critical Code Patterns

### 1. Data Loading (Adaptive Date Ranges)
**Pattern:** Adaptive date range based on timeframe to get ~700-1000 candles consistently.

```python
exchange = ccxt.binance({'enableRateLimit': True})  # CRITICAL: Rate limiting

# Adaptive lookback calculation
timeframe_minutes = timeframe_to_minutes(TIMEFRAME)
if timeframe_minutes <= 5:        # 1m, 5m → 3 days (~864 candles)
    lookback = datetime.now() - timedelta(days=3)
elif timeframe_minutes <= 30:     # 15m, 30m → 1 week
    lookback = datetime.now() - timedelta(weeks=1)
elif timeframe_minutes < 1440:    # 1h, 4h → 1 month
    lookback = datetime.now() - timedelta(days=30)
else:                              # 1d+ → 2 years
    lookback = datetime.now() - timedelta(days=730)

since = int(lookback.timestamp() * 1000)  # Milliseconds for Binance API

# Paginated fetching (max 1000 per request)
all_data = []
while True:
    data = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
    if len(data) == 0: break
    all_data.extend(data)
    since = data[-1][0] + 1  # Next candle timestamp
    if len(data) < 1000: break

# Helper function (required in all scripts):
def timeframe_to_minutes(tf_string):
    if tf_string.endswith('m'): return int(tf_string[:-1])
    elif tf_string.endswith('h'): return int(tf_string[:-1]) * 60
    elif tf_string.endswith('d'): return int(tf_string[:-1]) * 1440
    elif tf_string.endswith('w'): return int(tf_string[:-1]) * 10080
    else: return 60  # Default
```

### 2. LSTM Scaling Convention (NON-NEGOTIABLE)
**Critical:** Always use `[-1, 1]` range for LSTM. Save scaler with model.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))  # MUST be [-1, 1] for LSTM
df['close_scaled'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))

# CRITICAL: Persist scaler with model
torch.save(scaler, 'scaler.pth')

# Inverse transform predictions
predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
```

### 3. GPU Device Management (Notebooks + CUDA)
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"CUDA available: {torch.cuda.is_available()}")
if device == 'cuda':
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
model = LSTM().to(device)
X_train = X_train.to(device=device).float()  # Explicit dtype
```

### 4. OpenCL Platform Auto-Detection
**Pattern:** Score-based selection for best GPU platform.

```python
def detect_best_opencl_platform():
    """Scores: NVIDIA CUDA (100) > AMD ROCm (90) > Rusticl (70) > POCL (60)"""
    platforms = cl.get_platforms()
    best_score, best_platform, best_device = -1, None, None
    
    for platform in platforms:
        for device in platform.get_devices():
            score = 0
            pname = platform.name.lower()
            
            # Platform scoring
            if 'nvidia' in pname or 'cuda' in pname: score += 100
            elif 'amd' in pname or 'rocm' in pname: score += 90
            elif 'rusticl' in pname or 'mesa' in pname: score += 70
            
            # Device bonus
            if device.type == cl.device_type.GPU: score += 50
            score += min(device.max_compute_units, 50)
            
            if score > best_score:
                best_score, best_platform, best_device = score, platform, device
    
    return best_platform, best_device
```

### 5. Wavelet Data Caching (Parallel System)
**Purpose:** Eliminate redundant GPU computation across plot processes.

```python
# In prepare_data.py - compute once
data_dict = {
    'prices': prices,
    'approximations': approximations,  # 8 levels
    'details': details,                # 8 levels
    'currency': CURRENCY,
    'timeframe': TIMEFRAME
}
save_wavelet_data(CURRENCY, data_dict)  # → wavelet_plots/{currency}/wavelet_data.pkl

# In plot_*.py - load cached results
data = load_or_compute_wavelet_data(CURRENCY, TIMEFRAME)
prices = data['prices']
approximations = data['approximations']  # No recomputation!
```

### 6. Error Handling & Graceful Degradation
**Critical:** Scripts must handle missing dependencies (Pi Zero compatibility).

```python
# Pattern 1: Optional imports with fallback
CCXT_AVAILABLE = False
try:
    import ccxt
    CCXT_AVAILABLE = True
except:
    pass  # Fall back to demo mode

if '--demo' in sys.argv or not CCXT_AVAILABLE:
    # Generate synthetic data

# Pattern 2: Lazy matplotlib loading
MATPLOTLIB_AVAILABLE = False
plt = None  # Global

def init_matplotlib():
    global MATPLOTLIB_AVAILABLE, plt
    if MATPLOTLIB_AVAILABLE: return True
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt_module
        plt = plt_module
        MATPLOTLIB_AVAILABLE = True
        return True
    except Exception as e:
        print(f"⚠ Matplotlib error: {e}")
        print("  Run with --no-plots to skip visualization")
        return False
```

## Wavelet Transform Essentials

### Decomposition Semantics
- **Trend extraction:** `wavelet_level_filter(data, levels_to_filter=range(1,8))` → zeros details, keeps approximation
- **Volatility extraction:** `levels_to_filter=range(0,7)` → zeros approximation, keeps high-frequency details
- **Level semantics:** Level 0 = lowest frequency (trend), higher = noise/volatility
- **Wavelet types:** `'db4'` (smoothing), `'haar'` (edge detection), `'coif1'` (symmetric)
- **Depth constraint:** `len(data) >= 2^levels` (typically 5-9 levels)

### GPU Convolution Pattern (OpenCL)
```python
# OpenCL kernel (used in all GPU scripts)
convolution_kernel = """
__kernel void convolve(__global const float *signal,
                       __global const float *filter,
                       __global float *output,
                       const int sig_len, const int filt_len) {
    int i = get_global_id(0);
    if(i >= sig_len - filt_len + 1) return;
    float sum = 0.0f;
    for(int j = 0; j < filt_len; j++) {
        sum += signal[i + j] * filter[j];
    }
    output[i] = sum;
}
"""

# Apply with symmetric padding (matches PyWavelets 'symmetric' mode)
def gpu_convolve(signal, filter_coeffs, kernel, ctx, queue, mode='symmetric'):
    if mode == 'symmetric':
        pad_len = len(filter_coeffs) - 1
        signal_padded = symmetric_pad(signal, pad_len)
    # ... buffer ops ...
    return output[::2]  # Downsample by 2 for wavelet decomposition
```

### Anomaly Detection (MAD Threshold)
```python
def detect_anomalies_level(data, wavelet='haar', level=5, anomaly_levels=[0]):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    for lvl in anomaly_levels:
        coeff = coeffs[lvl]
        # Median Absolute Deviation threshold
        threshold = np.median(np.abs(coeff - np.median(coeff))) / 0.6745 * 3
        anomaly_indices = np.where(np.abs(coeff) > threshold)[0]
    return anomalies
```

## Development Workflows

### Adding Cryptocurrency Support
Change only these variables (then rerun cells in notebooks or pass as script args):
```python
symbol = 'MATIC/USDT'    # Any Binance pair
timeframe = '1h'         # Valid: '1m','5m','15m','30m','1h','4h','1d'
since_date = '2025-01-01T00:00:00Z'
limit = 1000             # API max, pagination handles more
```

### Environment Setup (Priority Order)
```bash
# Method 1: Automated (recommended)
./setup_env.sh  # Creates .venv, installs deps, detects GPU

# Method 2: Manual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Method 3: Notebook-based
# Uncomment %pip install lines in env.ipynb cells

# GPU-specific:
pip install pyopencl  # OpenCL (multi-platform)
# CUDA included in torch from PyTorch index
```

### Pi Zero Workarounds
```bash
# Problem: ccxt causes bus errors (memory/architecture issue)
# Solution 1: Demo mode (synthetic data)
python gpu_wavelet_cpu_plot.py --demo --no-plots

# Solution 2: Fetch data on another machine
# Transfer wavelet_data.pkl to Pi Zero, run plot scripts only
```

### Web Server Usage
```bash
# Start server (choose one)
python web_server_parallel.py   # Parallel (3.5x faster)
python web_server.py             # Sequential (simpler)

# Navigate to http://localhost:8080
# Select currency → Run Analysis
# Plots cached in wavelet_plots/{currency}/
```

### Testing & Verification
```bash
test_imports.py              # Validates all dependencies load
test_pywt_details.py         # PyWavelets installation check
test_wavelet_comparison.py   # CPU vs GPU accuracy verification
verify_symmetric_mode.py     # Boundary extension mode check
```

## Technical Constraints & Best Practices

### Notebook Execution Rules
1. **Sequential mandatory** - cells depend on kernel state (scaler, model, device)
2. **Scaler persistence** - always save with model: `torch.save(scaler, 'scaler.pth')`
3. **No modular imports** - notebooks are monolithic (all code inline)
4. **GPU state first** - set device before tensor operations

### Wavelet Limitations
- **Edge artifacts** - reconstruction has boundary effects at series start/end
- **Length constraint** - `len(data) >= 2^levels` for decomposition
- **Mode default** - PyWavelets uses 'symmetric' extension (affects boundaries)

### Multi-Platform GPU Priority
1. **OpenCL** (`gpu_wavelet_gpu_plot.py`) - auto-detects: NVIDIA CUDA → AMD ROCm → ARM Mali
2. **CUDA** (`gpu_wavelet_cuda_plot.py`) - NVIDIA only, auto CPU fallback
3. **Vulkan** (`gpu_wavelet_vulkan.py`) - compute shaders, most portable
4. **OpenGL** (`gpu_wavelet_opengl_plot.py`) - Raspberry Pi 5 VideoCore VII
5. **CPU** (`gpu_wavelet_cpu_plot.py`) - pure NumPy, Pi Zero safe

### Common Pitfalls
1. **Missing rate limiting** - `ccxt.binance({'enableRateLimit': True})` is mandatory
2. **Wrong scaler range** - LSTM requires `[-1, 1]`, not `[0, 1]`
3. **Incorrect padding** - must use 'symmetric' to match PyWavelets
4. **Downsampling omitted** - wavelets require `output[::2]` after convolution
5. **Pi Zero ccxt** - causes bus errors, always use `--demo` flag

## File Outputs & Caching

### Plot Generation (Per Run)
6 PNG files @ 300 DPI in `wavelet_plots/{currency}/`:
- `plot_01.png`: Main overview (price + 8-level decomposition)
- `plot_02a.png`: Progressive approximations (levels 0-8)
- `plot_02b.png`: Frequency band analysis
- `plot_03.png`: Anomaly detection overlay
- `plot_04.png`: Trading signal visualization
- `plot_05.png`: Statistical dashboard

### Cached Artifacts
- `wavelet_data.pkl`: Precomputed wavelets (used by parallel system)
- `metrics.json`: Performance stats (GPU time, data load time, etc.)

### Script Arguments Reference
```bash
# All GPU scripts accept:
python <script>.py <CURRENCY> <TIMEFRAME>
python gpu_wavelet_gpu_plot.py ETH 1h      # Currency + timeframe

# CPU script flags:
--demo        # Use synthetic data (no network)
--no-plots    # Skip matplotlib (computation only)

# Parallel system:
python prepare_data.py BTC 5m              # Data prep
python plot_01.py BTC 5m                   # Plot 1
python plot_template.py BTC 5m 02a         # Plot 2a (template)
```

## Project-Specific Conventions

### Import Pattern (All Scripts)
```python
# Standard order:
import numpy as np
import time
import sys
import os
from datetime import datetime, timedelta

# Optional with fallback:
CCXT_AVAILABLE = False
try:
    import ccxt
    CCXT_AVAILABLE = True
except: pass

# GPU (choose one per script):
import pyopencl as cl          # OpenCL
import torch                   # CUDA
from vulkan import *           # Vulkan (rare)
from OpenGL.GL import *        # OpenGL

# Lazy matplotlib:
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

### Argument Parsing (GPU Scripts)
```python
# Standard pattern across all GPU scripts:
CURRENCY = sys.argv[1].upper() if len(sys.argv) > 1 else 'BTC'
TIMEFRAME = sys.argv[2].lower() if len(sys.argv) > 2 else '5m'

# Validation:
if CURRENCY not in ['BTC', 'ETH', 'SOL']:
    print(f"Error: Unsupported currency '{CURRENCY}'")
    exit(1)

valid_timeframes = ['1m','5m','15m','30m','1h','4h','1d']
if TIMEFRAME not in valid_timeframes:
    print(f"Error: Unsupported timeframe")
    exit(1)
```

### Output Directory Pattern
```python
output_dir = f'wavelet_plots/{CURRENCY.lower()}'
os.makedirs(output_dir, exist_ok=True)
```

### Print Formatting
```python
print("=" * 70)
print(f"GPU-ACCELERATED WAVELET - {CURRENCY}/USDT ({TIMEFRAME})")
print("=" * 70)

print("[1/6] Loading data... ", end='', flush=True)
# ... work ...
print("✓")
```

## Key Dependencies

### Core (All Scripts)
- `numpy>=2.2.1` - numerical computing
- `pandas>=2.2.3` - data manipulation (notebooks)

### Data & ML (Notebooks)
- `ccxt>=4.4.37` - Binance API (causes Pi Zero bus errors!)
- `torch>=2.5.1` - PyTorch with CUDA
- `scikit-learn>=1.6.0` - MinMaxScaler, train_test_split

### Signal Processing
- `pywavelets>=1.7.0` - wavelet transforms (CPU reference)
- `scipy>=1.14.1` - interpolation (`interp1d` for GPU scripts)

### Visualization
- `matplotlib>=3.10.0` - plotting (may crash Pi Zero)
- `seaborn>=0.13.2` - statistical plots (notebooks)

### GPU Acceleration (Optional)
- `pyopencl>=2025.2.7` - OpenCL (multi-platform)
- `PyOpenGL>=3.1.7` - OpenGL compute (Pi 5 only)
- `glfw>=2.7.0` - OpenGL context (Pi 5)

### Installation by Use Case
```bash
# Full (all features):
pip install -r requirements.txt

# Minimal (CPU only):
pip install numpy pandas matplotlib

# Pi Zero (ultra-minimal):
pip install numpy
# Then: python gpu_wavelet_cpu_plot.py --demo --no-plots
```

