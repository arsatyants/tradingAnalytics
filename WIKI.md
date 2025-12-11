# Trading Analytics Wiki

Complete technical documentation for the trading analytics toolkit.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Pipeline](#data-pipeline)
3. [LSTM Price Prediction](#lstm-price-prediction)
4. [Wavelet Analysis](#wavelet-analysis)
5. [Technical Indicators](#technical-indicators)
6. [Anomaly Detection](#anomaly-detection)
7. [Visualization Guide](#visualization-guide)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Topics](#advanced-topics)

---

## Getting Started

### Environment Setup

#### Using env.ipynb
The `env.ipynb` notebook contains all dependency installation commands. Uncomment the lines you need:

```python
%pip install ccxt           # Cryptocurrency exchange API
%pip install pandas         # Data manipulation
%pip install torch          # Deep learning
%pip install numpy          # Numerical computing
%pip install scikit-learn   # ML utilities and scaling
%pip install matplotlib     # Plotting
%pip install seaborn        # Statistical visualization
%pip install mplfinance     # Financial charts
%pip install ta             # Technical analysis
%pip install pywavelets     # Wavelet transforms
%pip install scipy          # Scientific computing
%pip install statsmodels    # Statistical models
%pip install jupyterlab     # Notebook environment
%pip install onnx onnxscript # Model export
%pip install ssqueezepy     # Continuous wavelet transform
```

#### GPU Setup
To use CUDA acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA availability in notebook:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

---

## Data Pipeline

### Fetching Data from Binance

#### Basic Usage
```python
import ccxt
import pandas as pd

exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 100
since_date = '2025-10-01T00:00:00Z'

# Parse date to milliseconds
since = exchange.parse8601(since_date)

# Fetch OHLCV data
data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

# Convert to DataFrame
df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
```

#### Paginated Fetching
For large date ranges, implement pagination:

```python
def fetch_data_paginated(symbol, timeframe, since, limit):
    exchange = ccxt.binance()
    all_data = []
    
    while since < exchange.milliseconds():
        batch = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if len(batch) == 0:
            break
        since = batch[-1][0] + 1  # Move to next candle
        print(f"Fetched {len(batch)} candles")
        all_data += batch
        
    return all_data
```

### Data Preprocessing

#### Scaling for Neural Networks
**Critical**: LSTM models require scaled data in [-1, 1] range:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
df['close_scaled'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))
```

**Important**: Save the scaler with your model for production use:
```python
import torch
torch.save(scaler, 'scaler.pth')

# Load later
scaler = torch.load('scaler.pth')
```

#### Inverse Transformation
To convert predictions back to original scale:

```python
predictions_scaled = model_output.numpy()
predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
```

---

## LSTM Price Prediction

### Model Architecture

#### Standard LSTM Implementation
```python
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_layer_size=100, output_dim=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_dim, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_dim)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), 
            self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
```

### Training Process

#### Sequence Creation
```python
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
        
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)

# Example
train_seq_length = 12  # Use 12 candles to predict next
X_train, y_train = create_sequences(df['close_scaled'].values, train_seq_length)
```

#### Training Loop
```python
import torch.optim as optim

model = LSTM()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 150

for epoch in range(epochs):
    for seq, target in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        
        y_pred = model(seq)
        loss = loss_function(y_pred, target)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: loss = {loss.item()}')
```

### Multi-Step Prediction

#### Recursive Forecasting
```python
future_steps = 3
test_inputs = df['close_scaled'][-test_seq_length:].values.tolist()

model.eval()
with torch.no_grad():
    for _ in range(future_steps):
        seq = torch.FloatTensor(test_inputs[-test_seq_length:])
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        prediction = model(seq).item()
        test_inputs.append(prediction)

# Get predictions in original scale
predictions_scaled = np.array(test_inputs[-future_steps:]).reshape(-1, 1)
predictions = scaler.inverse_transform(predictions_scaled)
```

### Model Persistence

```python
# Save model
torch.save(model.state_dict(), 'lstm_model.pth')
torch.save(scaler, 'scaler.pth')

# Load model
model = LSTM()
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()
scaler = torch.load('scaler.pth')
```

---

## Wavelet Analysis

### Discrete Wavelet Transform (DWT)

#### Basic Decomposition
```python
import pywt

# Decompose signal into frequency bands
coeffs = pywt.wavedec(prices, wavelet='db4', level=5)

# coeffs[0] = approximation (trend)
# coeffs[1:] = details (high to low frequency)
```

#### Wavelet Selection Guide

| Wavelet | Best For | Characteristics |
|---------|----------|-----------------|
| `'haar'` | Sharp transitions | Simplest, fast computation |
| `'db4'`, `'db6'` | General time series | Good localization, smooth |
| `'coif1'` | Symmetric features | More symmetric than Daubechies |
| `'sym2'` | Balanced analysis | Near-symmetric, good for trends |

### Reconstruction and Filtering

#### Trend Extraction
Keep only the approximation coefficient (level 0):

```python
def extract_trend(time_series, wavelet='db4', level=7):
    coeffs = pywt.wavedec(time_series.flatten(), wavelet, level=level)
    
    # Zero out all detail coefficients
    reconstructed_coeffs = coeffs.copy()
    for i in range(1, len(reconstructed_coeffs)):
        reconstructed_coeffs[i] = np.zeros_like(reconstructed_coeffs[i])
    
    # Reconstruct
    trend = pywt.waverec(reconstructed_coeffs, wavelet)
    return trend
```

#### Volatility Extraction
Keep only detail coefficients (remove level 0):

```python
def extract_volatility(time_series, wavelet='haar', level=7):
    coeffs = pywt.wavedec(time_series.flatten(), wavelet, level=level)
    
    # Zero out approximation coefficient
    reconstructed_coeffs = coeffs.copy()
    reconstructed_coeffs[0] = np.zeros_like(reconstructed_coeffs[0])
    
    # Reconstruct
    volatility = pywt.waverec(reconstructed_coeffs, wavelet)
    return volatility
```

#### Custom Level Filtering
```python
def wavelet_level_filter(time_series, wavelet='db4', levels=5, 
                         levels_range_to_filter=range(1, 6)):
    coeffs = pywt.wavedec(time_series.flatten(), wavelet, level=levels)
    
    # Zero out specified levels
    reconstructed_coeffs = coeffs.copy()
    for i in levels_range_to_filter:
        reconstructed_coeffs[i] = np.zeros_like(reconstructed_coeffs[i])
    
    reconstructed = pywt.waverec(reconstructed_coeffs, wavelet)
    return reconstructed
```

### Continuous Wavelet Transform (CWT)

#### Time-Frequency Analysis
```python
from ssqueezepy import cwt
from ssqueezepy.visuals import imshow

# Apply CWT
Wx, scales = cwt(time_series, wavelet='hhhat')

# Visualize spectrogram
plt.figure(figsize=(14, 7))
imshow(Wx, abs=1, title='CWT Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
```

**Interpretation**:
- **X-axis**: Time progression
- **Y-axis**: Frequency bands (low to high)
- **Brightness**: Signal energy (bright = high activity)
- **Patterns**: Identify frequency bursts at specific moments

---

## Technical Indicators

### Stochastic Oscillator

#### Implementation
```python
def stochastic_oscillator(df, n=14, m=3):
    """
    Calculate stochastic oscillator %K and %D.
    
    Parameters:
    - n: Lookback period for %K (default 14)
    - m: Smoothing period for %D (default 3)
    """
    # Fast %K
    low_min = df['low'].rolling(n).min()
    high_max = df['high'].rolling(n).max()
    df['fast_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    
    # Fast %D (smoothed %K)
    df['fast_d'] = df['fast_k'].rolling(m).mean()
    
    # Slow %D (smoothed %D)
    df['slow_d'] = df['fast_d'].rolling(m).mean()
    
    return df
```

#### Trading Signals
```python
# Overbought/Oversold levels
overbought = 80
oversold = 20

# Generate signals
df['signal'] = 0
df.loc[df['fast_k'] < oversold, 'signal'] = 1   # Buy
df.loc[df['fast_k'] > overbought, 'signal'] = -1  # Sell
```

### Nadaraya-Watson Estimator

#### Kernel Regression
```python
from statsmodels.nonparametric.kernel_regression import KernelReg

def nadaraya_watson(x, y, bandwidth):
    """
    Apply Nadaraya-Watson kernel regression for smoothing.
    
    Parameters:
    - x: Independent variable (time index)
    - y: Dependent variable (prices)
    - bandwidth: Smoothing parameter (higher = smoother)
    """
    model = KernelReg(endog=y, exog=x, var_type='c', bw=[bandwidth])
    y_pred, _ = model.fit(x)
    return y_pred

# Usage
x = np.arange(len(df))
y = df['close'].values
bandwidth = 10
smoothed = nadaraya_watson(x, y, bandwidth)
```

#### Custom Implementation (from wave_nada.ipynb)
```python
import math

def nadaraya_watson_custom(data, h=8, mult=3):
    """
    Custom NW implementation with Gaussian kernel.
    
    Parameters:
    - h: Bandwidth parameter
    - mult: Multiplier for MAE bands
    """
    y = []
    sum_e = 0
    
    for i in range(len(data)):
        sum_val = 0
        sumw = 0
        
        for j in range(len(data)):
            w = math.exp(-(math.pow(i-j, 2) / (h*h*2)))
            sum_val += data[j] * w
            sumw += w
            
        y2 = sum_val / sumw
        sum_e += abs(data[i] - y2)
        y.append(y2)
    
    mae = sum_e / len(data) * mult
    
    # Calculate bands
    upper_band = [y[i] + mae for i in range(len(y))]
    lower_band = [y[i] - mae for i in range(len(y))]
    
    return y, upper_band, lower_band
```

---

## Anomaly Detection

### Wavelet-Based Detection

#### Algorithm Overview
1. Decompose signal using DWT
2. Calculate threshold using Median Absolute Deviation (MAD)
3. Apply soft thresholding to coefficients
4. Reconstruct signal
5. Compute residuals as anomalies

#### Implementation
```python
def detect_anomalies_level(data, wavelet='haar', level=5, anomaly_levels=[0]):
    """
    Detect anomalies in specific frequency bands.
    
    Parameters:
    - data: Price series
    - wavelet: Wavelet type
    - level: Decomposition depth
    - anomaly_levels: Which levels to analyze [0]=trend, [1,2,3]=volatility
    """
    anomalies = np.zeros(len(data))
    
    for lev in anomaly_levels:
        coeff = pywt.wavedec(data, wavelet, level=level)
        
        # MAD-based threshold
        threshold = (np.median(np.abs(coeff[lev])) / 0.6745 * 
                    np.sqrt(2 * np.log(len(data))) / 2)
        
        # Soft thresholding
        coeff[lev] = pywt.threshold(coeff[lev], value=threshold, mode='soft')
        
        # Reconstruct
        reconstructed = pywt.waverec(coeff, wavelet)
        
        # Trim to match original length
        if len(reconstructed) > len(data):
            reconstructed = reconstructed[:len(data)]
        
        # Accumulate anomalies
        anomalies += np.abs(data - reconstructed)
    
    return anomalies
```

#### Frequency Band Interpretation

| Level | Frequency Band | Detects |
|-------|---------------|---------|
| `[0]` | Trend/Approximation | Long-term deviations, regime changes |
| `[1]` | Highest frequency | Ultra-short volatility spikes |
| `[2]` | High frequency | Short-term volatility |
| `[3]` | Mid frequency | Medium-term volatility |
| `[1,2,3]` | Combined volatility | Comprehensive volatility anomalies |

#### Sensitivity Tuning

Adjust threshold calculation divisor for sensitivity:
```python
# More sensitive (detects smaller anomalies)
threshold = np.median(np.abs(coeff[lev])) / 0.5 * np.sqrt(2 * np.log(len(data))) / 2

# Less sensitive (only major anomalies)
threshold = np.median(np.abs(coeff[lev])) / 1.0 * np.sqrt(2 * np.log(len(data))) / 2
```

---

## Visualization Guide

### Price and Prediction Charts

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 10))

# Plot historical data
plt.plot(df.index, 
         scaler.inverse_transform(df['close'].values.reshape(-1, 1)), 
         label='Historical', color='blue')

# Plot predictions
future_dates = pd.date_range(
    start=df.index[-1], 
    periods=len(predictions) + 1, 
    freq='h'
)[1:]
plt.plot(future_dates, predictions, 
         label='Predicted', color='red', linewidth=2)

plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('BTC/USDT Price Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Wavelet Coefficient Visualization

```python
levels = 9
coeffs = pywt.wavedec(prices, 'db6', level=levels)

plt.figure(figsize=(20, levels * 3))

# Plot approximation
plt.subplot(levels + 1, 1, 1)
plt.plot(coeffs[0], color='cyan')
plt.title(f'Approximation Coefficients (cA{levels})')
plt.grid(True)

# Plot details
for lv in range(1, levels + 1):
    plt.subplot(levels + 1, 1, lv + 1)
    plt.plot(coeffs[lv], color='cyan')
    plt.title(f'Detail Coefficients (cD{lv})')
    plt.grid(True)

plt.tight_layout()
plt.show()
```

### Multi-Panel Technical Charts

```python
fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Price panel
ax[0].plot(df.index, df['close'], color='blue', linewidth=0.8)
ax[0].set_ylabel('Price [$]')
ax[0].grid(True)

# Stochastic Oscillator panel
ax[1].plot(df.index, df['fast_k'], color='orange', label='%K')
ax[1].plot(df.index, df['fast_d'], color='grey', label='%D')
ax[1].plot(df.index, df['slow_d'], color='green', label='Slow %D')
ax[1].axhline(y=80, color='red', linestyle='--', label='Overbought')
ax[1].axhline(y=20, color='green', linestyle='--', label='Oversold')
ax[1].set_ylabel('Stochastic Oscillator')
ax[1].set_ylim(0, 100)
ax[1].legend()
ax[1].grid(True)

plt.xlabel('Date')
plt.tight_layout()
plt.show()
```

---

## API Reference

### Key Functions

#### `load_data(symbol, timeframe, limit, since_date)`
Fetch cryptocurrency data from Binance.

**Parameters:**
- `symbol` (str): Trading pair (e.g., 'BTC/USDT')
- `timeframe` (str): Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
- `limit` (int): Candles per API request
- `since_date` (str): Start date in ISO format

**Returns:** `(DataFrame, MinMaxScaler)` - Price data and fitted scaler

#### `wavelet_level_filter(time_series, wavelet, levels, levels_range_to_filter)`
Apply wavelet filtering to specific frequency bands.

**Parameters:**
- `time_series` (np.array): Input signal
- `wavelet` (str): Wavelet type ('db4', 'haar', etc.)
- `levels` (int): Decomposition depth
- `levels_range_to_filter` (range): Levels to zero out

**Returns:** `np.array` - Reconstructed filtered signal

#### `detect_anomalies_level(data, wavelet, level, anomaly_levels)`
Detect anomalies using wavelet analysis.

**Parameters:**
- `data` (np.array): Price series
- `wavelet` (str): Wavelet type
- `level` (int): Decomposition depth
- `anomaly_levels` (list): Frequency bands to analyze

**Returns:** `np.array` - Anomaly scores

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size or model size
hidden_layer_size = 50  # Instead of 100

# Or use CPU
device = 'cpu'
```

#### Scaler Mismatch
```
ValueError: X has 1 features but MinMaxScaler is expecting N features
```

**Solution:** Always use same scaler for prediction:
```python
# Save with model
torch.save(scaler, 'scaler.pth')

# Load before prediction
scaler = torch.load('scaler.pth')
```

#### Wavelet Length Mismatch
```python
# Trim reconstructed signal to match original
if len(reconstructed) > len(original):
    reconstructed = reconstructed[:len(original)]
```

#### API Rate Limits
Add delays between requests:
```python
import time

for batch in batches:
    data = exchange.fetch_ohlcv(...)
    time.sleep(1)  # 1 second delay
```

---

## Advanced Topics

### Hyperparameter Optimization

#### Grid Search for LSTM
```python
param_grid = {
    'hidden_size': [50, 100, 200],
    'seq_length': [12, 24, 48],
    'learning_rate': [0.001, 0.01, 0.1]
}

best_loss = float('inf')
best_params = None

for hidden in param_grid['hidden_size']:
    for seq_len in param_grid['seq_length']:
        for lr in param_grid['learning_rate']:
            model = LSTM(hidden_layer_size=hidden)
            # Train and evaluate
            # Update best_params if loss improved
```

### Model Export

#### ONNX Format
```python
import torch.onnx

# Export trained model
dummy_input = torch.randn(12, 1, 1)
torch.onnx.export(model, dummy_input, "lstm_model.onnx")
```

### Real-Time Streaming

```python
while True:
    # Fetch latest candle
    latest = exchange.fetch_ohlcv(symbol, timeframe, limit=1)
    
    # Update DataFrame
    new_row = pd.DataFrame(latest, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = pd.concat([df, new_row]).tail(1000)  # Keep last 1000
    
    # Make prediction
    # ... (prediction code)
    
    time.sleep(60)  # Wait for next candle
```

### Ensemble Methods

Combine LSTM with wavelets:
```python
# LSTM prediction
lstm_pred = model(sequence)

# Wavelet trend
trend = extract_trend(recent_prices)
trend_direction = 1 if trend[-1] > trend[-2] else -1

# Combined signal
final_pred = lstm_pred * 0.7 + trend_direction * 0.3
```

---

## Performance Benchmarks

Typical execution times on different hardware:

| Task | CPU (i7) | GPU (RTX 3080) |
|------|----------|----------------|
| LSTM Training (150 epochs) | ~5 min | ~30 sec |
| Wavelet Decomposition (5 levels) | ~1 sec | N/A |
| CWT Spectrogram | ~3 sec | N/A |
| Anomaly Detection | ~2 sec | N/A |

---

## Best Practices

1. **Always save scalers** with models for deployment
2. **Use GPU** for LSTM training when available
3. **Validate predictions** with historical backtesting
4. **Monitor API limits** from exchanges
5. **Version control models** with timestamp and parameters
6. **Document experiments** in notebook markdown cells
7. **Test on multiple timeframes** before production use

---

**Last Updated**: December 2025  
**Maintained by**: @arsatyants
