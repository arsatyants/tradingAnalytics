# Trading Analytics - AI Agent Instructions

## Project Overview
Cryptocurrency trading toolkit for BTC/ETH/SOL price prediction and anomaly detection using LSTM networks and wavelet transforms. Real-time data from Binance via CCXT. Features multi-platform GPU acceleration and web interface.

## Architecture & Components

### Core Files
- **btc-prediction.ipynb**: LSTM prediction + CWT spectrograms + anomaly detection (BTC)
- **wave_nada.ipynb**: DWT decomposition + Stochastic Oscillator + Nadaraya-Watson smoothing (SOL)
- **gpu_wavelet_gpu_plot.py**: OpenCL wavelet acceleration (NVIDIA/AMD/ARM Mali) - generates 6 PNG plots per currency
- **gpu_wavelet_gpu_console.py**: Console-only version with ASCII graphs + period/amplitude analysis
- **gpu_wavelet_cpu_plot.py**: Pure NumPy fallback with `--demo` mode (Pi Zero compatible, no ccxt/matplotlib)
- **gpu_wavelet_cuda_plot.py**: PyTorch/CUDA acceleration with automatic CPU fallback
- **web_server.py**: HTTP server at localhost:8080 for interactive browser-based analysis

### Script Execution Patterns
- **Single currency**: `python gpu_wavelet_gpu_plot.py BTC 5m` (currency + timeframe args)
- **All currencies**: `./run_all_currencies.sh` generates 18 PNG files (BTC/ETH/SOL × 6 plots each)
- **Console mode**: `./run_all_currencies_console.sh` for terminal output with ASCII graphs
- **Web interface**: `python web_server.py` → http://localhost:8080
- **Demo mode**: `python gpu_wavelet_cpu_plot.py --demo --no-plots` (Pi Zero safe - no network/matplotlib)

## Critical Code Patterns

### Data Loading (Mandatory Pattern for All Files)
```python
exchange = ccxt.binance({'enableRateLimit': True})  # Rate limiting critical
since = exchange.parse8601('2024-12-01T00:00:00Z')
# Paginated fetching for large ranges
all_data = []
while since < exchange.milliseconds():
    data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    if len(data) == 0: break
    since = data[-1][0] + 1  # Move to next candle timestamp
    all_data += data

df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
```

### Scaling Convention (LSTM REQUIREMENT)
```python
scaler = MinMaxScaler(feature_range=(-1, 1))  # ALWAYS [-1, 1] for LSTM
df['close_scaled'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))
# CRITICAL: Save scaler with model: torch.save(scaler, 'scaler.pth')
predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
```

### Device Management (Notebooks + CUDA Scripts)
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"CUDA available: {torch.cuda.is_available()}")
if device == 'cuda':
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
model = LSTM().to(device)
X_train = X_train.to(device=device).float()
```

### Wavelet Decomposition Semantics
- **Trend extraction**: `wavelet_level_filter(data, wavelet='db4', levels=7, levels_to_filter=range(1,8))` - zeros out details, keeps approximation (level 0)
- **Volatility extraction**: `levels_to_filter=range(0,7)` - zeros out approximation, keeps high-frequency details
- **Standard wavelets**: `'db4'` (smoothing), `'db6'` (sharper), `'haar'` (edge detection), `'coif1'` (symmetric)
- **Level semantics**: Level 0 = lowest frequency (trend), higher levels = increasing frequencies (noise/volatility)
- **Typical depth**: 5-9 levels depending on data length (must satisfy `len(data) >= 2^levels`)

### Wavelet Filter Implementation (From wave_nada.ipynb)
```python
def wavelet_level_filter(time_series, wavelet='db4', levels=5, levels_to_filter=None):
    if levels_to_filter is None:
        levels_to_filter = [1, 2, 3, 4, 5]
    _series = time_series.reshape(-1, 1)
    coeffs = pywt.wavedec(_series.flatten(), wavelet, level=levels)
    filtered_coeffs = coeffs.copy()
    for i in levels_to_filter:
        filtered_coeffs[i] = np.zeros_like(filtered_coeffs[i])  # Zero out unwanted levels
    reconstructed = pywt.waverec(filtered_coeffs, wavelet)
    return reconstructed[:len(time_series)]  # Truncate to original length
```

### Anomaly Detection (MAD Threshold Method)
```python
def detect_anomalies_level(data, wavelet='haar', level=5, anomaly_levels=[0]):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    anomalies = np.zeros(len(data), dtype=bool)
    for lvl in anomaly_levels:
        coeff = coeffs[lvl]
        # MAD = Median Absolute Deviation
        threshold = np.median(np.abs(coeff - np.median(coeff))) / 0.6745 * 3
        # Mark anomalies where coefficient exceeds threshold
        anomaly_indices = np.where(np.abs(coeff) > threshold)[0]
    return anomalies
```

## Development Workflows

### Adding New Cryptocurrency Support
Only these variables need changing (rerun notebook cells):
```python
symbol = 'MATIC/USDT'    # Or any Binance pair: BTC/USDT, ETH/USDT, SOL/USDT, etc.
timeframe = '1h'         # Valid: '1m', '5m', '15m', '30m', '1h', '4h', '1d'
since_date = '2025-01-01T00:00:00Z'
limit = 1000             # Max 1000 per request (pagination loop handles more)
```

### GPU Script Arguments
```bash
# Supported currencies: BTC, ETH, SOL
python gpu_wavelet_gpu_plot.py BTC 5m      # Currency + timeframe
python gpu_wavelet_gpu_console.py ETH 1h   # Console version
python gpu_wavelet_cpu_plot.py --demo      # Synthetic data (no network)
python gpu_wavelet_cpu_plot.py --demo --no-plots  # Computation only (Pi Zero)
```

### Environment Setup Priority
```bash
# Method 1: Automated script (recommended)
./setup_env.sh  # Creates .venv, installs all deps, detects GPU

# Method 2: Manual venv
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Method 3: Notebook-based (use env.ipynb)
# Uncomment %pip install lines in env.ipynb cells

# GPU-specific additions:
pip install pyopencl  # For OpenCL (multi-platform)
# CUDA already included in torch package from PyTorch index
```

### Raspberry Pi Zero Workaround
```bash
# Problem: ccxt causes bus errors on Pi Zero (memory/architecture issue)
# Solution: Use demo mode to bypass ccxt entirely
python gpu_wavelet_cpu_plot.py --demo --no-plots
# Generates synthetic BTC data (87000 ± 3000 with noise)
# Skips matplotlib to avoid crashes on low-memory systems
```

### Web Server Usage
```bash
python web_server.py
# Navigate to http://localhost:8080
# Select currency (BTC/ETH/SOL) → Click "Run Analysis"
# Server runs gpu_wavelet_gpu_plot.py subprocess
# Displays 6 generated plots in browser
# Plots cached in wavelet_plots/{currency}/ directory
```

## Technical Constraints

### Notebook Execution Rules
- **Sequential execution mandatory** - cells depend on kernel state (scaler, model, device)
- **Scaler persistence required** - save alongside model: `torch.save(scaler, 'scaler.pth')`
- **No modular imports** - all code is inline (notebooks are monolithic, not library-based)
- **GPU state management** - device must be set before tensor operations begin

### Wavelet Limitations
- **Edge artifacts** - reconstruction may have boundary effects at series start/end
- **Length requirements** - `len(data) >= 2^levels` for decomposition level
- **Mode defaults** - PyWavelets uses 'symmetric' extension mode (affects boundary handling)

### Multi-Platform GPU Support
- **OpenCL (gpu_wavelet_gpu_plot.py)** - auto-detects best platform: NVIDIA CUDA → AMD ROCm → ARM Mali → Intel
- **CUDA (gpu_wavelet_cuda_plot.py)** - NVIDIA only, auto-falls back to CPU if unavailable
- **OpenGL (gpu_wavelet_opengl_plot.py)** - compute shaders for Raspberry Pi 5 VideoCore VII
- **CPU fallback (gpu_wavelet_cpu_plot.py)** - pure NumPy, no dependencies beyond numpy/pandas

## Error Handling & Testing

### Common Import Failures
Scripts handle `ccxt` and `matplotlib` import failures gracefully:
```python
CCXT_AVAILABLE = False
try:
    import ccxt
    CCXT_AVAILABLE = True
except: pass  # Fall back to --demo mode

if '--demo' in sys.argv or not CCXT_AVAILABLE:
    # Generate synthetic data instead
```

### Verification Scripts
- **test_imports.py** - validates all dependencies load correctly
- **test_pywt_details.py** - verifies PyWavelets installation and wavelet availability
- **test_wavelet_comparison.py** - compares CPU vs GPU wavelet outputs for accuracy
- **verify_symmetric_mode.py** - checks PyWavelets boundary extension mode
