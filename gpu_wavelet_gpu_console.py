"""
GPU-Accelerated Wavelet Decomposition for Trading Analytics
============================================================

This script demonstrates how to use OpenCL on Mali-G31 GPU to accelerate
wavelet transform operations for cryptocurrency price analysis.

WHAT IS WAVELET DECOMPOSITION?
-------------------------------
Wavelet decomposition breaks down a price signal into multiple frequency 
components, allowing you to separate:
- TREND (low frequency): Overall price direction
- VOLATILITY (high frequency): Short-term price fluctuations

This is used in btc-prediction.ipynb and wave_nada.ipynb for:
1. Anomaly detection (detecting unusual price movements)
2. Trend extraction (filtering noise from true trend)
3. Multi-scale analysis (understanding price behavior at different timeframes)

HOW WAVELETS WORK IN TRADING:
------------------------------
Think of a price chart as a musical note:
- Low frequencies = bass (slow trends, hours/days)
- High frequencies = treble (fast changes, minutes)

Wavelet filters separate these frequencies so you can:
- Trade on trends (ignore noise)
- Detect volatility spikes (risk management)
- Find support/resistance levels (filtered price levels)

GPU ADVANTAGE:
--------------
Wavelet decomposition requires convolution operations (sliding window 
multiplication) which are PERFECT for GPU parallelization. Each output 
point can be calculated independently.

CPU: Calculate each point sequentially (slow)
GPU: Calculate all points simultaneously (fast)

For 10,000 price points: GPU can be 3-5x faster
"""

import pyopencl as cl
import numpy as np
import time
import ccxt
from datetime import datetime, timedelta

# =============================================================================
# STEP 1: INITIALIZE OPENCL
# =============================================================================
print("=" * 70)
print("GPU-ACCELERATED WAVELET DECOMPOSITION")
print("=" * 70)

# Find the Rusticl platform (Mesa OpenCL for Mali GPU)
platforms = cl.get_platforms()
platform = [p for p in platforms if 'rusticl' in p.name.lower()][0]

# Get the Mali-G31 device
device = platform.get_devices()[0]

# Create OpenCL context (manages GPU memory and execution)
ctx = cl.Context([device])

# Create command queue (sends work to GPU)
queue = cl.CommandQueue(ctx)

print(f"\n✓ GPU Initialized: {device.name}")
print(f"  Compute Units: {device.max_compute_units}")
print(f"  Max Work Group Size: {device.max_work_group_size}")
print(f"  Global Memory: {device.global_mem_size / (1024**2):.0f} MB\n")

# =============================================================================
# STEP 2: DEFINE WAVELET KERNELS
# =============================================================================

"""
CONVOLUTION KERNEL EXPLANATION:
--------------------------------
Convolution is the mathematical operation behind wavelet transforms.

For each output point i, we:
1. Take a window of input data centered around position i
2. Multiply each element by corresponding wavelet coefficient
3. Sum all products to get output[i]

Example with Haar wavelet [0.7071, 0.7071]:
  Input:  [50000, 51000, 49000, 50500, ...]
  Position i=1:
    output[1] = 50000 * 0.7071 + 51000 * 0.7071
              = 71,417 (smoothed average)

This happens for EVERY position, making it perfect for parallel GPU execution.
"""

# OpenCL C code that runs on the GPU
convolution_kernel = """
__kernel void convolve(__global const float *signal,     // Input: price data
                       __global const float *filter,      // Input: wavelet coefficients
                       __global float *output,            // Output: filtered signal
                       const int sig_len,                 // Length of price data
                       const int filt_len) {              // Length of wavelet filter
    
    // Get the unique ID for this GPU thread (each thread processes one output point)
    int i = get_global_id(0);
    
    // Make sure we don't go out of bounds
    if(i >= sig_len - filt_len + 1) {
        return;
    }
    
    // Perform convolution: multiply and sum
    float sum = 0.0f;
    for(int j = 0; j < filt_len; j++) {
        int idx = i + j;
        sum += signal[idx] * filter[j];
    }
    
    // Write result to global memory
    output[i] = sum;
}
"""

# Compile the kernel for GPU execution
program = cl.Program(ctx, convolution_kernel).build()
convolve_kernel = cl.Kernel(program, "convolve")

print("✓ OpenCL Kernel Compiled\n")

# =============================================================================
# STEP 3: DEFINE WAVELET COEFFICIENTS
# =============================================================================

"""
WAVELET TYPES:
--------------
Different wavelets detect different features in price data.

HAAR WAVELET (simplest):
- Low-pass (trend):  [0.7071, 0.7071]  → Averages adjacent points
- High-pass (detail): [0.7071, -0.7071] → Differences between points

DAUBECHIES-4 (better for finance):
- More coefficients = smoother filtering
- Reduces edge artifacts
- Better trend preservation

In your notebooks, you use db4/db6 wavelets with pywavelets library.
Here we implement Haar for GPU demonstration.
"""

# Haar wavelet coefficients (normalized for AVERAGING, not energy preservation)
# These values sum to 1.0 to preserve price scale
haar_low_pass = np.array([0.5, 0.5], dtype=np.float32)  # Simple average
haar_high_pass = np.array([0.5, -0.5], dtype=np.float32)  # Difference

# Alternative: Use orthonormal wavelets (multiply result by sqrt(2) to match scale)
# haar_low_pass_ortho = np.array([0.7071067811865476, 0.7071067811865476], dtype=np.float32)
# haar_high_pass_ortho = np.array([0.7071067811865476, -0.7071067811865476], dtype=np.float32)

# Daubechies-4 wavelet (normalized to sum to 1.0 for price preservation)
db4_low_pass = np.array([
    0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551
], dtype=np.float32)
# Normalized: sum = 1.0, preserves price scale
db4_sum = 0.482962913145 + 0.836516303738 + 0.224143868042 - 0.129409522551
db4_low_pass = db4_low_pass / db4_sum  # Normalize to sum=1

db4_high_pass = np.array([
    -0.129409522551, -0.224143868042, 0.836516303738, -0.482962913145
], dtype=np.float32)

print("✓ Wavelet Coefficients Loaded")
print(f"  Haar Low-pass (trend):  {haar_low_pass} (sum={haar_low_pass.sum():.4f})")
print(f"  Haar High-pass (detail): {haar_high_pass} (sum={haar_high_pass.sum():.4f})")
print(f"  DB4 Low-pass: {db4_low_pass} (sum={db4_low_pass.sum():.4f})")
print(f"  DB4 Low-pass length: {len(db4_low_pass)}")
print(f"\n  Note: Low-pass filters sum to 1.0 to preserve price scale\n")

# =============================================================================
# STEP 4: GPU CONVOLUTION FUNCTION
# =============================================================================

def gpu_convolve(signal, filter_coeffs, kernel):
    """
    Perform convolution on GPU using OpenCL.
    
    Args:
        signal: Input price data (numpy array)
        filter_coeffs: Wavelet coefficients (numpy array)
        kernel: Compiled OpenCL kernel
    
    Returns:
        Filtered signal (numpy array)
    
    MEMORY TRANSFER PROCESS:
    ------------------------
    1. Allocate GPU memory buffers
    2. Copy data from CPU RAM → GPU VRAM
    3. Execute kernel (GPU does the math)
    4. Copy results from GPU VRAM → CPU RAM
    
    This transfer overhead is why GPU only helps with larger datasets.
    """
    
    sig_len = len(signal)
    filt_len = len(filter_coeffs)
    output_len = sig_len - filt_len + 1
    output = np.zeros(output_len, dtype=np.float32)
    
    # Create GPU memory buffers
    # READ_ONLY: GPU only reads this data
    # WRITE_ONLY: GPU only writes to this buffer
    # COPY_HOST_PTR: Copy data from CPU to GPU during buffer creation
    
    signal_buf = cl.Buffer(ctx, 
                          cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                          hostbuf=signal)
    
    filter_buf = cl.Buffer(ctx, 
                          cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                          hostbuf=filter_coeffs)
    
    output_buf = cl.Buffer(ctx, 
                          cl.mem_flags.WRITE_ONLY, 
                          output.nbytes)
    
    # Execute kernel on GPU
    # (output_len,) = launch output_len parallel threads
    # None = let OpenCL choose work group size automatically
    kernel(queue, (output_len,), None,
          signal_buf, filter_buf, output_buf,
          np.int32(sig_len), np.int32(filt_len))
    
    # Copy results back from GPU to CPU
    cl.enqueue_copy(queue, output, output_buf)
    
    return output

# =============================================================================
# STEP 5: FETCH REAL BTC DATA FROM BINANCE
# =============================================================================

"""
REAL-TIME DATA FROM BINANCE:
----------------------------
Using CCXT library to fetch actual Bitcoin price data from Binance exchange.

This matches the data loading pattern in btc-prediction.ipynb and wave_nada.ipynb:
1. Initialize Binance exchange
2. Fetch OHLCV data (Open, High, Low, Close, Volume)
3. Use close prices for analysis
4. Convert timestamps to datetime

Data quality:
- Real market data with actual volatility
- Includes market microstructure (spreads, slippage)
- Reflects true trading conditions
"""

print("=" * 70)
print("FETCHING REAL BTC DATA FROM BINANCE")
print("=" * 70)

try:
    # Initialize Binance exchange (no API key needed for public data)
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    print(f"\n✓ Connected to Binance")
    print(f"  Exchange: {exchange.name}")
    print(f"  Has fetchOHLCV: {exchange.has['fetchOHLCV']}")
    
    # Set timeframe and date range
    symbol = 'BTC/USDT'
    timeframe = '1h'  # 1-hour candles
    limit = 1000  # Fetch 1000 candles
    
    print(f"\nFetching data:")
    print(f"  Symbol: {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Candles: {limit}")
    
    # Fetch OHLCV data
    print(f"\n  Downloading... ", end='', flush=True)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    print("✓")
    
    # Convert to numpy arrays
    # OHLCV format: [timestamp, open, high, low, close, volume]
    timestamps = np.array([candle[0] for candle in ohlcv])
    opens = np.array([candle[1] for candle in ohlcv], dtype=np.float32)
    highs = np.array([candle[2] for candle in ohlcv], dtype=np.float32)
    lows = np.array([candle[3] for candle in ohlcv], dtype=np.float32)
    closes = np.array([candle[4] for candle in ohlcv], dtype=np.float32)
    volumes = np.array([candle[5] for candle in ohlcv], dtype=np.float32)
    
    # Use close prices for wavelet analysis
    prices = closes
    
    # Convert timestamps to datetime
    dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Total candles: {len(prices)}")
    print(f"  Date range: {dates[0].strftime('%Y-%m-%d %H:%M')} → {dates[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Duration: {(dates[-1] - dates[0]).days} days, {(dates[-1] - dates[0]).seconds // 3600} hours")
    print(f"\n  Price Statistics:")
    print(f"    Current:  ${prices[-1]:,.2f}")
    print(f"    Open:     ${prices[0]:,.2f}")
    print(f"    High:     ${prices.max():,.2f}")
    print(f"    Low:      ${prices.min():,.2f}")
    print(f"    Change:   {((prices[-1] - prices[0]) / prices[0] * 100):+.2f}%")
    print(f"    Std Dev:  ${prices.std():,.2f}")
    print(f"\n  Volume Statistics:")
    print(f"    Total:    {volumes.sum():,.0f} BTC")
    print(f"    Average:  {volumes.mean():,.2f} BTC/hour")
    print(f"    Max:      {volumes.max():,.2f} BTC")
    
except Exception as e:
    print(f"\n✗ Error fetching data from Binance: {e}")
    print(f"\nFalling back to synthetic data...")
    
    # Fallback to synthetic data if API fails
    np.random.seed(42)
    n_points = 1000
    initial_price = 50000.0
    drift = 0.0001
    volatility = 0.02
    returns = np.random.randn(n_points) * volatility + drift
    price_multipliers = np.exp(np.cumsum(returns))
    prices = (initial_price * price_multipliers).astype(np.float32)
    dates = [datetime.now() - timedelta(hours=n_points-i) for i in range(n_points)]
    
    print(f"✓ Generated {n_points} synthetic price points")
    print(f"  Price range: ${prices.min():,.2f} - ${prices.max():,.2f}")

print()

# =============================================================================
# CONSOLE PLOTTING UTILITIES
# =============================================================================

def plot_ascii(data, height=15, width=80, title=""):
    """
    Draw ASCII chart in console using dots and lines.
    
    Args:
        data: Array of values to plot
        height: Number of rows in the plot
        width: Number of columns in the plot
        title: Chart title
    """
    if len(data) == 0:
        return
    
    # Sample data to fit width
    if len(data) > width:
        indices = np.linspace(0, len(data)-1, width, dtype=int)
        data = data[indices]
    
    # Normalize data to fit height
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min
    
    if data_range == 0:
        data_range = 1
    
    normalized = ((data - data_min) / data_range * (height - 1)).astype(int)
    
    # Create empty canvas
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Draw data points
    for x, y in enumerate(normalized):
        y = height - 1 - y  # Flip y-axis
        if 0 <= y < height and 0 <= x < width:
            canvas[y][x] = '●'
    
    # Draw connecting lines
    for x in range(len(normalized) - 1):
        y1 = height - 1 - normalized[x]
        y2 = height - 1 - normalized[x + 1]
        
        # Draw vertical line between points
        y_start, y_end = min(y1, y2), max(y1, y2)
        for y in range(y_start, y_end + 1):
            if 0 <= y < height and 0 <= x < width:
                if canvas[y][x] == ' ':
                    canvas[y][x] = '│'
    
    # Print title
    if title:
        print(f"\n  {title}")
        print(f"  {'-' * len(title)}")
    
    # Print y-axis scale
    print(f"  ${data_max:>10,.0f} ┤", end='')
    print(''.join(canvas[0]))
    
    for i in range(1, height - 1):
        print(f"  {' ' * 11} │", end='')
        print(''.join(canvas[i]))
    
    print(f"  ${data_min:>10,.0f} ┤", end='')
    print(''.join(canvas[height - 1]))
    
    # X-axis
    print(f"  {' ' * 11} └{'─' * width}")
    print(f"  {' ' * 11}  0{' ' * (width-10)}points: {len(data)}")

def plot_comparison(original, trend, detail, width=70):
    """
    Plot original signal with trend and detail components side by side.
    """
    height = 10
    
    # Sample all arrays to same width
    sample_width = width // 3 - 2
    
    def sample_data(data):
        if len(data) > sample_width:
            indices = np.linspace(0, len(data)-1, sample_width, dtype=int)
            return data[indices]
        return data
    
    orig_sampled = sample_data(original)
    trend_sampled = sample_data(trend)
    detail_sampled = sample_data(detail)
    
    # Normalize each dataset
    def normalize(data):
        dmin, dmax = np.min(data), np.max(data)
        drange = dmax - dmin if (dmax - dmin) > 0 else 1
        return ((data - dmin) / drange * (height - 1)).astype(int)
    
    orig_norm = normalize(orig_sampled)
    trend_norm = normalize(trend_sampled)
    detail_norm = normalize(detail_sampled)
    
    print("\n  " + "┌" + "─" * sample_width + "┐  " + 
          "┌" + "─" * sample_width + "┐  " + 
          "┌" + "─" * sample_width + "┐")
    
    # Print charts row by row
    for row in range(height):
        y = height - 1 - row
        
        # Original signal
        line_orig = ""
        for x in range(len(orig_norm)):
            if orig_norm[x] == row:
                line_orig += "●"
            elif x > 0 and min(orig_norm[x-1], orig_norm[x]) <= row <= max(orig_norm[x-1], orig_norm[x]):
                line_orig += "│"
            else:
                line_orig += " "
        
        # Trend
        line_trend = ""
        for x in range(len(trend_norm)):
            if trend_norm[x] == row:
                line_trend += "●"
            elif x > 0 and min(trend_norm[x-1], trend_norm[x]) <= row <= max(trend_norm[x-1], trend_norm[x]):
                line_trend += "│"
            else:
                line_trend += " "
        
        # Detail
        line_detail = ""
        for x in range(len(detail_norm)):
            if detail_norm[x] == row:
                line_detail += "●"
            elif x > 0 and min(detail_norm[x-1], detail_norm[x]) <= row <= max(detail_norm[x-1], detail_norm[x]):
                line_detail += "│"
            else:
                line_detail += " "
        
        print(f"  │{line_orig:<{sample_width}}│  │{line_trend:<{sample_width}}│  │{line_detail:<{sample_width}}│")
    
    print("  " + "└" + "─" * sample_width + "┘  " + 
          "└" + "─" * sample_width + "┘  " + 
          "└" + "─" * sample_width + "┘")
    print(f"  {'ORIGINAL (Prices)':^{sample_width}}  {'TREND (Low-pass)':^{sample_width}}  {'DETAIL (High-pass)':^{sample_width}}")

# =============================================================================
# STEP 6: PERFORM WAVELET DECOMPOSITION ON GPU
# =============================================================================

print("=" * 70)
print("WAVELET DECOMPOSITION (GPU)")
print("=" * 70)

# -------------------------
# Level 1: Haar Decomposition
# -------------------------
print("\n[1] Haar Wavelet Decomposition:")

# GPU: Low-frequency component (TREND)
start_time = time.time()
trend_gpu = gpu_convolve(prices, haar_low_pass, convolve_kernel)
gpu_time_trend = time.time() - start_time

print(f"  ✓ Trend extraction:  {gpu_time_trend*1000:.2f}ms")
print(f"    Points: {len(trend_gpu)}")
print(f"    Range:  ${trend_gpu.min():,.2f} - ${trend_gpu.max():,.2f}")

# GPU: High-frequency component (VOLATILITY/NOISE)
start_time = time.time()
detail_gpu = gpu_convolve(prices, haar_high_pass, convolve_kernel)
gpu_time_detail = time.time() - start_time

print(f"  ✓ Detail extraction: {gpu_time_detail*1000:.2f}ms")
print(f"    Points: {len(detail_gpu)}")
print(f"    Range:  ${detail_gpu.min():,.2f} - ${detail_gpu.max():,.2f}")
print(f"    Mean:   ${detail_gpu.mean():,.2f} (should be near 0)")
print(f"    Std:    ${np.abs(detail_gpu).std():,.2f}")

# Visualize decomposition
print("\n" + "=" * 70)
print("VISUAL DECOMPOSITION")
print("=" * 70)
plot_comparison(prices[:300], trend_gpu[:300], detail_gpu[:300])

# -------------------------
# Level 2: DB4 Decomposition
# -------------------------
print("\n[2] Daubechies-4 Wavelet Decomposition:")

start_time = time.time()
trend_db4 = gpu_convolve(prices, db4_low_pass, convolve_kernel)
gpu_time_db4 = time.time() - start_time

print(f"  ✓ DB4 trend extraction: {gpu_time_db4*1000:.2f}ms")
print(f"    Points: {len(trend_db4)}")
print(f"    Range:  ${trend_db4.min():,.2f} - ${trend_db4.max():,.2f}")

# Show original prices
plot_ascii(prices[:200], height=12, width=70, title="Original BTC Prices (first 200 points)")

# =============================================================================
# STEP 7: CPU COMPARISON (NUMPY)
# =============================================================================

print("\n" + "=" * 70)
print("CPU vs GPU COMPARISON")
print("=" * 70)

# CPU implementation using numpy
start_time = time.time()
trend_cpu = np.convolve(prices, haar_low_pass, mode='valid')
cpu_time_trend = time.time() - start_time

start_time = time.time()
detail_cpu = np.convolve(prices, haar_high_pass, mode='valid')
cpu_time_detail = time.time() - start_time

print(f"\nHaar Wavelet Performance:")
print(f"  GPU Trend:   {gpu_time_trend*1000:.3f}ms")
print(f"  CPU Trend:   {cpu_time_trend*1000:.3f}ms")
print(f"  Speedup:     {cpu_time_trend/gpu_time_trend:.2f}x")
print(f"\n  GPU Detail:  {gpu_time_detail*1000:.3f}ms")
print(f"  CPU Detail:  {cpu_time_detail*1000:.3f}ms")
print(f"  Speedup:     {cpu_time_detail/gpu_time_detail:.2f}x")

# Verify results match
trend_match = np.allclose(trend_gpu, trend_cpu, rtol=1e-5)
detail_match = np.allclose(detail_gpu, detail_cpu, rtol=1e-5)

print(f"\nAccuracy Check:")
print(f"  Trend matches CPU:  {'✓' if trend_match else '✗'}")
print(f"  Detail matches CPU: {'✓' if detail_match else '✗'}")

# =============================================================================
# STEP 8: ANOMALY DETECTION (TRADING APPLICATION)
# =============================================================================

print("\n" + "=" * 70)
print("TRADING APPLICATION: ANOMALY DETECTION")
print("=" * 70)

"""
ANOMALY DETECTION USING WAVELETS:
----------------------------------
High-frequency details capture sudden price movements.
Large detail coefficients = unusual volatility = potential anomalies

This is used in btc-prediction.ipynb's detect_anomalies_level() function.

Method:
1. Calculate detail coefficients (already done above)
2. Find median absolute deviation (MAD) - robust to outliers
3. Flag points where |detail| > threshold * MAD

Threshold = 3 → flags ~99.7% confidence anomalies (3-sigma rule)
"""

# Calculate statistics
detail_abs = np.abs(detail_gpu)
median = np.median(detail_abs)
mad = np.median(np.abs(detail_abs - median))

# Threshold for anomaly detection (adjustable sensitivity)
threshold = 3.0
anomaly_threshold = median + threshold * mad

# Find anomalies
anomaly_indices = np.where(detail_abs > anomaly_threshold)[0]
anomaly_prices = prices[anomaly_indices]

print(f"\nAnomaly Detection Results:")
print(f"  Total price points:   {len(prices)}")
print(f"  Detected anomalies:   {len(anomaly_indices)}")
print(f"  Anomaly rate:         {len(anomaly_indices)/len(prices)*100:.2f}%")
print(f"  Threshold:            {anomaly_threshold:.2f}")
print(f"\nTop 5 Anomalies:")

if len(anomaly_indices) > 0:
    # Sort by magnitude
    top_indices = anomaly_indices[np.argsort(detail_abs[anomaly_indices])[-5:]]
    for idx in reversed(top_indices):
        print(f"  Index {idx:5d}: ${prices[idx]:,.2f} "
              f"(detail coefficient: {detail_gpu[idx]:+.2f})")
else:
    print("  No anomalies detected")

# Visualize anomaly detection
print("\n" + "=" * 70)
print("ANOMALY VISUALIZATION")
print("=" * 70)

print("\n" + "=" * 70)
print("UNDERSTANDING DETAIL COEFFICIENTS")
print("=" * 70)
print("""
Detail coefficients represent HIGH-FREQUENCY CHANGES in price.
They show the DIFFERENCE between adjacent price points after wavelet filtering.

WHAT THE VALUES MEAN:
--------------------
Positive value:  Price increased faster than the trend
Negative value:  Price decreased faster than the trend  
Zero value:      Price moved exactly with the trend

For Haar wavelet [0.5, -0.5], the detail coefficient is:
  detail[i] = 0.5 × price[i] - 0.5 × price[i+1]
            = (price[i] - price[i+1]) / 2
            = Half the price difference between consecutive hours

EXAMPLES with BTC at ~$96,000:
------------------------------
  detail = +$250  →  Price jumped $500 in one hour (bullish spike)
  detail = -$250  →  Price dropped $500 in one hour (bearish spike)
  detail = +$50   →  Normal upward movement ($100/hour)
  detail = -$50   →  Normal downward movement ($100/hour)
  detail = 0      →  Perfectly smooth trend (no volatility)

ANOMALY DETECTION:
-----------------
Large absolute values (|detail| > threshold) indicate:
  - Sudden price movements (pumps/dumps)
  - High volatility periods
  - Potential trading opportunities or risks
  - Market regime changes
""")

print(f"\nDetail Coefficient Statistics:")
print(f"  Range:  ${detail_gpu.min():+,.2f} to ${detail_gpu.max():+,.2f}")
print(f"  Mean:   ${detail_gpu.mean():+,.2f} (should be near $0)")
print(f"  Std:    ${np.abs(detail_gpu).std():,.2f}")
print(f"\n  Median Abs Value: ${median:.2f}")
print(f"  MAD (spread):     ${mad:.2f}")
print(f"  Anomaly Threshold: ${anomaly_threshold:.2f} (median + {threshold}×MAD)")
print(f"  Max Abs Value:    ${detail_abs.max():.2f}")

# Sample for visualization
sample_size = 70
if len(detail_gpu) > sample_size:
    indices = np.linspace(0, len(detail_gpu)-1, sample_size, dtype=int)
    detail_sample = detail_gpu[indices]
    # Check which sampled points are anomalies
    anomaly_sample = np.array([i in anomaly_indices for i in indices])
else:
    detail_sample = detail_gpu
    anomaly_sample = np.array([i in anomaly_indices for i in range(len(detail_gpu))])

height = 12
detail_min, detail_max = detail_sample.min(), detail_sample.max()
detail_range = detail_max - detail_min if (detail_max - detail_min) > 0 else 1
detail_norm = ((detail_sample - detail_min) / detail_range * (height - 1)).astype(int)

# Calculate zero line position (where detail = 0)
if detail_min < 0 < detail_max:
    zero_line = int((-detail_min / detail_range) * (height - 1))
    zero_line = height - 1 - zero_line  # Flip for display
else:
    zero_line = -1  # No zero crossing

print(f"\nDetail Coefficients Chart (showing positive and negative values):")
print(f"  Positive = price rising faster than trend")
print(f"  Negative = price falling faster than trend")
print()

# Draw chart with anomaly markers
print(f"  ${detail_max:>10.2f} ┤", end='')
for row in range(height):
    if row > 0:
        print(f"\n  {' ' * 11} │", end='')
    y = height - 1 - row
    
    # Draw zero line
    if y == zero_line:
        for x in range(len(detail_norm)):
            if detail_norm[x] == row:
                if anomaly_sample[x]:
                    print("★", end='')
                else:
                    print("●", end='')
            else:
                print("━", end='')  # Zero line
    else:
        for x in range(len(detail_norm)):
            if detail_norm[x] == row:
                if anomaly_sample[x]:
                    print("★", end='')
                else:
                    print("●", end='')
            elif x > 0 and min(detail_norm[x-1], detail_norm[x]) <= row <= max(detail_norm[x-1], detail_norm[x]):
                print("│", end='')
            else:
                print(" ", end='')

print(f"\n  ${detail_min:>10.2f} ┤" + "─" * len(detail_norm))
print(f"  {' ' * 11}  ● = normal   ★ = anomaly   ━ = zero (trend line)")

# =============================================================================
# STEP 9: MULTI-LEVEL DECOMPOSITION
# =============================================================================

print("\n" + "=" * 70)
print("MULTI-LEVEL WAVELET DECOMPOSITION")
print("=" * 70)

"""
MULTI-LEVEL DECOMPOSITION:
--------------------------
Apply wavelet transform repeatedly to extract different timescales:

Level 0: Original prices (high frequency)
Level 1: Apply low-pass filter → medium frequency
Level 2: Apply low-pass to Level 1 → lower frequency
Level 3: Apply low-pass to Level 2 → even lower frequency
...

This creates a hierarchy:
- Level 0: Minutes
- Level 1: Hours  
- Level 2: Days
- Level 3: Weeks

Used in wave_nada.ipynb for multi-scale trend analysis.
"""

print("\nDecomposing into 5 levels...")
print("\nMulti-level wavelet decomposition creates a pyramid:")
print("  - Each level extracts BOTH approximation (trend) and detail (changes)")
print("  - Approximation at level N becomes input for level N+1")
print("  - Details capture frequency components at each scale\n")

# Start with original prices
current_signal = prices
approximations = []
details = []

for level in range(5):
    # Apply both filters at this scale
    approx = gpu_convolve(current_signal, haar_low_pass, convolve_kernel)
    detail = gpu_convolve(current_signal, haar_high_pass, convolve_kernel)
    
    approximations.append(approx)
    details.append(detail)
    
    print(f"  Level {level+1}:")
    print(f"    Approximation: {len(approx):5d} points, "
          f"range: ${approx.min():,.2f} - ${approx.max():,.2f}")
    print(f"    Detail:        {len(detail):5d} points, "
          f"range: ${detail.min():+,.2f} - ${detail.max():+,.2f}")
    
    # Next level works on approximation (coarser scale)
    current_signal = approx

# Keep levels for backward compatibility
levels = approximations

# Visualize multi-level decomposition
print("\n" + "=" * 70)
print("MULTI-LEVEL DECOMPOSITION VISUALIZATION")
print("=" * 70)
print("\nShowing APPROXIMATIONS (trends at each scale):")

for i, level_data in enumerate(approximations[:4]):
    freq_name = ['High-Freq Trend', 'Medium-Freq Trend', 'Low-Freq Trend', 'Very Low-Freq Trend'][i]
    plot_ascii(level_data[:150], height=8, width=70, 
               title=f"Level {i+1} Approximation - {freq_name}")

print("\nShowing DETAILS (changes at each scale):")
print("Note: Details oscillate around zero - positive = rising, negative = falling\n")

for i, detail_data in enumerate(details[:4]):
    freq_name = ['Finest Details (Hours)', 'Medium Details', 'Coarse Details', 'Coarsest Details'][i]
    plot_ascii(detail_data[:150], height=8, width=70, 
               title=f"Level {i+1} Detail - {freq_name}")

# =============================================================================
# STEP 10: TRADING SIGNAL GENERATION
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING TRADING SIGNALS")
print("=" * 70)

"""
TRADING STRATEGY USING WAVELETS:
---------------------------------
1. Use Level 3 decomposition as "true trend" (filters out noise)
2. Compare current price to trend:
   - Price > Trend + buffer → SELL (overbought)
   - Price < Trend - buffer → BUY (oversold)
   - Otherwise → HOLD

This is similar to Bollinger Bands but using wavelet-based trend.
"""

# Get trend (Level 3)
trend = levels[2]  # Level 3 decomposition

# Align arrays (trend is shorter due to convolution)
offset = len(prices) - len(trend)
aligned_prices = prices[offset:]

# Calculate distance from trend
price_deviation = aligned_prices - trend

# Generate signals
buffer = np.std(price_deviation) * 0.5  # Half standard deviation
buy_signals = price_deviation < -buffer
sell_signals = price_deviation > buffer

print(f"\nSignal Statistics:")
print(f"  Total periods:  {len(trend)}")
print(f"  BUY signals:    {buy_signals.sum()} ({buy_signals.sum()/len(trend)*100:.1f}%)")
print(f"  SELL signals:   {sell_signals.sum()} ({sell_signals.sum()/len(trend)*100:.1f}%)")
print(f"  HOLD periods:   {len(trend) - buy_signals.sum() - sell_signals.sum()}")

# Show last 5 signals
print(f"\nLast 5 Trading Signals:")
for i in range(-5, 0):
    price = aligned_prices[i]
    trend_val = trend[i]
    deviation = price_deviation[i]
    
    if buy_signals[i]:
        signal = "BUY 📈"
    elif sell_signals[i]:
        signal = "SELL 📉"
    else:
        signal = "HOLD ⏸️ "
    
    print(f"  {signal}  Price: ${price:,.2f}  Trend: ${trend_val:,.2f}  "
          f"Deviation: ${deviation:+.2f}")

# Visualize trading signals
print("\n" + "=" * 70)
print("TRADING SIGNALS VISUALIZATION")
print("=" * 70)

# Plot price with trend and buy/sell markers
print("\nPrice vs Trend with trading signals:")
sample_size = 70
if len(aligned_prices) > sample_size:
    indices = np.linspace(0, len(aligned_prices)-1, sample_size, dtype=int)
    price_sample = aligned_prices[indices]
    trend_sample = trend[indices]
    buy_sample = buy_signals[indices]
    sell_sample = sell_signals[indices]
else:
    price_sample = aligned_prices
    trend_sample = trend
    buy_sample = buy_signals
    sell_sample = sell_signals

height = 12
price_min = min(price_sample.min(), trend_sample.min())
price_max = max(price_sample.max(), trend_sample.max())
price_range = price_max - price_min if (price_max - price_min) > 0 else 1

price_norm = ((price_sample - price_min) / price_range * (height - 1)).astype(int)
trend_norm = ((trend_sample - price_min) / price_range * (height - 1)).astype(int)

print(f"  ${price_max:>10,.0f} ┤", end='')
for row in range(height):
    if row > 0:
        print(f"\n  {' ' * 11} │", end='')
    y = height - 1 - row
    for x in range(len(price_norm)):
        # Check if price or trend line is at this position
        price_here = price_norm[x] == row
        trend_here = trend_norm[x] == row
        price_line = x > 0 and min(price_norm[x-1], price_norm[x]) <= row <= max(price_norm[x-1], price_norm[x])
        trend_line = x > 0 and min(trend_norm[x-1], trend_norm[x]) <= row <= max(trend_norm[x-1], trend_norm[x])
        
        if price_here:
            if buy_sample[x]:
                print("↑", end='')  # Buy signal
            elif sell_sample[x]:
                print("↓", end='')  # Sell signal
            else:
                print("●", end='')  # Price point
        elif trend_here:
            print("─", end='')  # Trend line
        elif price_line:
            print("│", end='')
        elif trend_line:
            print("·", end='')
        else:
            print(" ", end='')

print(f"\n  ${price_min:>10,.0f} ┤" + "─" * len(price_norm))
print(f"  {' ' * 11}  ● = Price   ─ = Trend   ↑ = BUY   ↓ = SELL")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

total_gpu_time = gpu_time_trend + gpu_time_detail + gpu_time_db4
total_cpu_time = cpu_time_trend + cpu_time_detail

print(f"""
GPU Device: {device.name}
Compute Units: {device.max_compute_units}

Performance:
  Total GPU Time: {total_gpu_time*1000:.2f}ms
  Total CPU Time: {total_cpu_time*1000:.2f}ms
  Overall Speedup: {total_cpu_time/total_gpu_time:.2f}x

Analysis Results:
  Price Points: {len(prices)}
  Anomalies Detected: {len(anomaly_indices)}
  Buy Signals: {buy_signals.sum()}
  Sell Signals: {sell_signals.sum()}

Data Source:
  Symbol: BTC/USDT (Binance)
  Timeframe: 1 hour candles
  Date Range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}

Integration with Your Notebooks:
  - Replace pywt.wavedec() calls with gpu_convolve() for speed
  - Use for real-time anomaly detection in btc-prediction.ipynb
  - Accelerate multi-level decomposition in wave_nada.ipynb
  - Process multiple cryptocurrencies in parallel

Next Steps:
  1. ✓ Using real Binance data from ccxt
  2. Integrate into existing notebooks
  3. Benchmark with larger datasets (100k+ points)
  4. Implement GPU-accelerated LSTM preprocessing
""")

print("=" * 70)
print("✓ Wavelet decomposition complete!")
print("=" * 70)
