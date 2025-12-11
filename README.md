# Trading Analytics

A cryptocurrency trading analytics toolkit for price prediction and anomaly detection using deep learning (LSTM networks) and advanced signal processing techniques (wavelet transforms). Features real-time data analysis for Bitcoin and Solana with interactive Jupyter notebooks.

## 🚀 Features

- **LSTM Price Prediction**: Multi-step ahead forecasting using PyTorch LSTM networks
- **Wavelet Analysis**: Multi-resolution decomposition for trend/volatility separation
- **Anomaly Detection**: Automated detection of price anomalies across frequency bands
- **Technical Indicators**: Stochastic Oscillator and Nadaraya-Watson kernel regression
- **Real-Time Data**: Live cryptocurrency price feeds from Binance via CCXT
- **GPU Acceleration**: CUDA support for faster model training and inference

## 📊 Supported Analysis

| Notebook | Cryptocurrency | Key Features |
|----------|---------------|--------------|
| `btc-prediction.ipynb` | BTC/USDT | LSTM prediction, wavelet filtering, CWT spectrograms, anomaly detection |
| `wave_nada.ipynb` | SOL/USDT | Wavelet decomposition, Stochastic Oscillator, Nadaraya-Watson smoothing |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for acceleration)
- Jupyter Lab/Notebook

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/arsatyants/tradingAnalytics.git
cd tradingAnalytics
```

2. **Install dependencies**

Open `env.ipynb` and uncomment the installation commands you need, or install via pip:

```bash
pip install ccxt pandas torch numpy scikit-learn matplotlib seaborn \
            mplfinance ta pywavelets scipy statsmodels jupyterlab \
            onnx onnxscript ssqueezepy
```

3. **Launch Jupyter Lab**
```bash
jupyter lab
```

4. **Open a notebook** and start with the data loading cells

## 📖 Usage

### BTC Price Prediction (btc-prediction.ipynb)

```python
# Configure parameters
symbol = 'BTC/USDT'
timeframe = '1h'  # Options: '15m', '1h', '4h', '1d'
since_date = '2025-10-01T00:00:00Z'

# Load data and train LSTM
df, scaler = load_data(symbol, timeframe, limit=100, since_date=since_date)

# Model will predict next 3 time steps
future_steps = 3
```

### Wavelet Analysis (both notebooks)

```python
# Trend extraction (filter out high-frequency noise)
reconstructed_trend = wavelet_level_filter(
    time_series, 
    wavelet='db4', 
    levels=7, 
    levels_range_to_filter=range(1, 8)  # Keep level 0 only
)

# Volatility extraction (filter out trend)
reconstructed_volatility = wavelet_level_filter(
    time_series, 
    wavelet='haar', 
    levels=7, 
    levels_range_to_filter=range(0, 7)  # Keep high-frequency details
)
```

### Anomaly Detection

```python
# Detect anomalies in specific frequency bands
anomalies = detect_anomalies_level(
    df['close'], 
    wavelet='haar', 
    level=5, 
    anomaly_levels=[1, 2, 3]  # Analyze volatility bands
)
```

## 🧠 Architecture

### Data Pipeline
```
Binance API (CCXT) → Paginated OHLCV Fetch → Pandas DataFrame → 
MinMaxScaler [-1, 1] → LSTM/Wavelet Processing → Predictions/Analysis
```

### Key Components

1. **Data Acquisition**: Real-time and historical data via `ccxt.binance()`
2. **Preprocessing**: MinMaxScaler normalization to [-1, 1] range
3. **LSTM Model**: Custom PyTorch implementation with configurable architecture
4. **Wavelet Engine**: PyWavelets library for multi-level decomposition
5. **Visualization**: Matplotlib/Seaborn for comprehensive charting

## 📐 Model Architecture

### LSTM Configuration
- **Input dimension**: 1 (close price)
- **Hidden layer size**: 100 units
- **Output dimension**: 1 (predicted price)
- **Lookback window**: Configurable (`train_seq_length`, `test_seq_length`)
- **Prediction horizon**: 3 steps ahead (default)

### Wavelet Parameters
- **Wavelets used**: Daubechies (`db4`, `db6`), Haar, Coiflet (`coif1`)
- **Decomposition levels**: 5-9 levels (adjust based on data length)
- **Threshold method**: Soft thresholding with MAD-based threshold calculation

## 🔧 Configuration

### Switching Cryptocurrencies
```python
# Edit in any notebook
symbol = 'ETH/USDT'  # or 'SOL/USDT', 'BNB/USDT', etc.
timeframe = '15m'    # Adjust timeframe as needed
since_date = '2025-11-01T00:00:00Z'
```

### GPU/CPU Selection
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
```

### LSTM Hyperparameters
```python
hidden_layer_size = 100  # Increase for more complex patterns
train_seq_length = 12    # Lookback window size
future_steps = 3         # Prediction horizon
```

## 📊 Visualizations

The notebooks generate multiple visualization types:
- **Price charts**: Original vs. predicted prices with datetime indexing
- **Wavelet coefficients**: Multi-level frequency band decomposition
- **CWT spectrograms**: Time-frequency analysis showing activity patterns
- **Anomaly overlays**: Detected anomalies highlighted on price data
- **Technical indicators**: Stochastic Oscillator with overbought/oversold zones

## 🔬 Methodology

### LSTM Prediction
1. Fetch historical OHLCV data from Binance
2. Scale prices to [-1, 1] range
3. Create sequences for supervised learning
4. Train LSTM on GPU/CPU
5. Generate multi-step predictions
6. Inverse transform to original price scale

### Wavelet Anomaly Detection
1. Perform discrete wavelet transform (DWT)
2. Calculate MAD-based threshold per level
3. Apply soft thresholding to coefficients
4. Reconstruct signal and compute residuals
5. Identify anomalies as significant deviations

### Trend/Volatility Separation
- **Trend**: Keep approximation coefficients (level 0), zero out details
- **Volatility**: Zero out approximation, keep detail coefficients
- **Reconstruction**: Inverse wavelet transform for each component

## 📁 Project Structure

```
tradingAnalytics/
├── btc-prediction.ipynb    # BTC analysis: LSTM + wavelets + anomalies
├── wave_nada.ipynb         # SOL analysis: wavelets + technical indicators
├── env.ipynb               # Dependency installation helper
├── .github/
│   └── copilot-instructions.md  # AI agent development guide
├── README.md               # This file
└── LICENSE                 # Project license
```

## 🎯 Use Cases

- **Day Trading**: Identify short-term price movements and volatility patterns
- **Risk Management**: Detect anomalies and unusual market behavior
- **Technical Analysis**: Generate smoothed indicators and trend signals
- **Research**: Experiment with wavelets and deep learning for time series

## ⚠️ Limitations

- **No backtesting framework**: Predictions are forward-only (no historical validation)
- **Sequential execution**: Notebook cells must be run in order
- **Scaler dependency**: Saved models require corresponding scaler for deployment
- **Edge artifacts**: Wavelet reconstruction may have boundary effects
- **Live trading**: This is an analysis tool, not a trading bot

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Automated backtesting framework
- Additional technical indicators
- More cryptocurrency pairs
- Model hyperparameter optimization
- Real-time streaming data support

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 👤 Author

**Arsa Tyants** ([@arsatyants](https://github.com/arsatyants))

## 🙏 Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) for unified cryptocurrency exchange API
- [PyWavelets](https://pywavelets.readthedocs.io/) for wavelet transform library
- [PyTorch](https://pytorch.org/) for deep learning framework
- [ssqueezepy](https://github.com/OverLordGoldDragon/ssqueezepy) for CWT analysis

## 📚 Further Reading

- [Wavelet Analysis for Time Series](https://en.wikipedia.org/wiki/Wavelet_transform)
- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Technical Analysis Indicators](https://www.investopedia.com/terms/t/technicalindicator.asp)

---

**Disclaimer**: This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk. Always conduct your own research and never invest more than you can afford to lose.
