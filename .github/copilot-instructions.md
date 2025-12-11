# Trading Analytics - AI Agent Instructions

## Project Overview
Cryptocurrency trading analytics toolkit focused on BTC and SOL price prediction and anomaly detection using deep learning (LSTM) and signal processing techniques (wavelet transforms). All analysis is done in Jupyter notebooks with real-time data from Binance via CCXT.

## Architecture & Data Flow

### Core Components
1. **Data Acquisition** (`btc-prediction.ipynb`, `wave_nada.ipynb`)
   - Live data fetching from Binance using `ccxt.binance()`
   - Historical OHLCV data loaded in batches via `fetch_ohlcv()`
   - Data normalized with `MinMaxScaler(feature_range=(-1, 1))` for neural network compatibility

2. **LSTM Prediction Pipeline** (`btc-prediction.ipynb`)
   - Custom LSTM model for multi-step price forecasting (default: 3 steps ahead)
   - GPU acceleration via CUDA when available
   - Model persistence: `lstm_model.pth`, `scaler.pth`

3. **Wavelet Analysis** (both notebooks)
   - Multi-level wavelet decomposition for trend/volatility separation
   - Anomaly detection using soft thresholding on detail coefficients
   - Continuous Wavelet Transform (CWT) for time-frequency analysis

4. **Technical Indicators** (`wave_nada.ipynb`)
   - Stochastic Oscillator (%K, %D) with custom implementation
   - Nadaraya-Watson kernel regression for smoothing

## Critical Patterns & Conventions

### Data Loading Pattern
All data loading follows this structure:
```python
exchange = ccxt.binance()
since = exchange.parse8601('YYYY-MM-DDTHH:MM:SSZ')
# Paginated fetching until exchange.milliseconds()
df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
```

### Scaling Convention
**Critical**: All price data must be scaled to `[-1, 1]` range before LSTM processing. Original prices recovered via `scaler.inverse_transform()` for visualization/output.

### Wavelet Decomposition Levels
- **Trend extraction**: Filter out levels 1 to N (keep level 0 approximation)
- **Volatility extraction**: Filter out level 0 (keep high-frequency details)
- Common wavelets: `'db4'`, `'db6'`, `'haar'`, `'coif1'`
- Typical decomposition depth: 5-9 levels

### Device Management
Always check CUDA availability:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Move tensors/models to device
```

## Common Workflows

### Adding New Cryptocurrency Analysis
1. Update `symbol` variable (e.g., `'ETH/USDT'`, `'SOL/USDT'`)
2. Adjust `timeframe` (`'1h'`, `'15m'`, `'1d'`)
3. Set `since_date` for historical range
4. Rerun data loading cell - no code changes needed

### Modifying LSTM Architecture
Key parameters in model definition:
- `input_dim`: Number of features (default: 1 for close price only)
- `hidden_layer_size`: LSTM hidden units (default: 100)
- `output_dim`: Prediction dimension (default: 1)
- `train_seq_length`, `test_seq_length`: Lookback windows

### Anomaly Detection Tuning
Adjust sensitivity via:
- `level` parameter in `detect_anomalies_level()`: higher = broader patterns
- `anomaly_levels` list: which frequency bands to analyze `[0]` (trend), `[1,2,3]` (volatility bands)
- Threshold calculation uses MAD (Median Absolute Deviation): modify divisor `0.6745` for sensitivity

## Dependencies & Environment

### Installation Order (from `env.ipynb`)
Core dependencies must be installed in this sequence:
```
ccxt, pandas, torch, numpy, scikit-learn, matplotlib, seaborn, 
mplfinance, ta, pywavelets, scipy, statsmodels, jupyterlab, 
onnx, onnxscript, ssqueezepy
```

**Note**: `env.ipynb` contains commented installation commands - uncomment selectively as needed.

## Visualization Patterns

### Multi-Panel Price Charts
Use `plt.figure(figsize=(14,10))` for clarity. When plotting predictions, extend index with:
```python
future_dates = pd.date_range(start=df.index[-1], periods=len(predictions) + 1, freq='h')[1:]
```

### Wavelet Coefficient Plots
Stack decomposition levels vertically with `plt.subplot(levels+1, 1, i)` for visual comparison across frequency bands.

## Known Limitations
- No automated backtesting framework - predictions are forward-only
- Scaler must be saved/loaded with model for deployment
- LSTM requires sequential execution - cells cannot be run out of order
- Wavelet reconstruction may introduce edge artifacts at series boundaries

## File Relationships
- `btc-prediction.ipynb`: Complete BTC analysis (LSTM + wavelets + anomalies)
- `wave_nada.ipynb`: SOL-focused with Nadaraya-Watson and Stochastic Oscillator
- `env.ipynb`: Dependency installation helper
- No Python modules - all code is notebook-native
