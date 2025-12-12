# Trading Analytics - AI Agent Instructions

## Project Overview
Cryptocurrency trading toolkit for BTC/SOL price prediction and anomaly detection using LSTM networks and wavelet transforms. Real-time data from Binance via CCXT.

## File Structure
- **btc-prediction.ipynb**: LSTM prediction + wavelet filtering + anomaly detection (BTC)
- **wave_nada.ipynb**: Wavelet analysis + Stochastic Oscillator + Nadaraya-Watson smoothing (SOL)
- **gpu_wavelet_cuda_plot.py**: PyTorch/CUDA wavelet acceleration with auto CPU fallback
- **gpu_wavelet_gpu_plot.py**: OpenCL wavelet acceleration (multi-platform: NVIDIA/AMD/ARM Mali)
- **gpu_wavelet_opengl_plot.py**: OpenGL compute shader acceleration (Raspberry Pi 5 / VideoCore VII)

## Critical Patterns

### Data Loading (all notebooks/scripts use this pattern)
```python
exchange = ccxt.binance()
since = exchange.parse8601('2024-12-01T00:00:00Z')
data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)  # Paginated loop
df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
```

### Scaling Convention (REQUIRED for LSTM)
```python
scaler = MinMaxScaler(feature_range=(-1, 1))  # Always [-1, 1] range
df['close'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))
# Recover: scaler.inverse_transform(predictions)
```

### Device Management
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTM().to(device)
tensor = tensor.to(device)
```

### Wavelet Analysis
- **Trend extraction**: `levels_to_filter=[1,2,3,4,5]` (keep level 0 approximation)
- **Volatility extraction**: `levels_to_filter=[0]` (keep high-frequency details)
- Common wavelets: `'db4'`, `'db6'`, `'haar'`, `'coif1'`
- Decomposition depth: 5-9 levels typical

### Anomaly Detection
```python
anomalies = detect_anomalies_level(data, wavelet='haar', level=5, anomaly_levels=[0])
# anomaly_levels: [0]=trend anomalies, [1,2,3]=volatility bands
# Threshold uses MAD: np.median(np.abs(coeff)) / 0.6745
```

## Development Workflow

### Adding New Cryptocurrency
Only change these variables, rerun cells:
```python
symbol = 'ETH/USDT'      # Or 'SOL/USDT', 'BTC/USDT'
timeframe = '1h'          # Options: '15m', '1h', '4h', '1d'
since_date = '2025-01-01T00:00:00Z'
```

### Running GPU Scripts
```bash
python gpu_wavelet_cuda_plot.py    # CUDA/PyTorch (auto CPU fallback)
python gpu_wavelet_gpu_plot.py     # OpenCL (multi-platform GPU)
python gpu_wavelet_opengl_plot.py  # OpenGL compute (Raspberry Pi 5)
# Outputs saved to wavelet_plots_cuda/, wavelet_plots/, or wavelet_plots_opengl/
```

### Raspberry Pi 5 Setup (OpenGL)
```bash
sudo apt install libglfw3 libglfw3-dev mesa-utils
pip install PyOpenGL PyOpenGL_accelerate glfw
python gpu_wavelet_opengl_plot.py  # Uses VideoCore VII GPU
```

### Environment Setup
```bash
pip install -r requirements.txt
# Optional GPU: pip install pyopencl (for OpenCL scripts)
```

## Key Constraints
- Notebooks must run cells **sequentially** - kernel state dependencies
- Scaler must be saved alongside model for inference
- Wavelet reconstruction may have edge artifacts at series boundaries
- All code is notebook-native - no reusable Python modules
