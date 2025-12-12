"""
GPU-Accelerated Wavelet Decomposition with OpenGL Compute Shaders
=================================================================

This script uses OpenGL compute shaders for GPU acceleration, specifically
designed for Raspberry Pi 5's VideoCore VII GPU.

Features:
- OpenGL 4.3+ / OpenGL ES 3.1+ compute shader support
- High-resolution plots (300 DPI)
- Multiple subplots for comprehensive analysis
- Real BTC data from Binance
- Automatic fallback to CPU if OpenGL compute unavailable
- Saves all plots to 'wavelet_plots_opengl/' directory

Raspberry Pi 5 Support:
- VideoCore VII GPU supports OpenGL ES 3.1 compute shaders
- Uses Mesa drivers with V3D backend
- Requires: sudo apt install python3-opengl libglfw3 libglfw3-dev

Installation:
    pip install PyOpenGL PyOpenGL_accelerate glfw numpy
    # On Raspberry Pi OS:
    sudo apt install libglfw3 libglfw3-dev mesa-utils
"""

import numpy as np
import time
import os
from datetime import datetime, timedelta

# Try to import OpenGL components
OPENGL_AVAILABLE = False
try:
    import glfw
    from OpenGL.GL import *
    from OpenGL.GL import shaders
    OPENGL_AVAILABLE = True
except ImportError as e:
    print(f"⚠ OpenGL not available: {e}")
    print("  Install with: pip install PyOpenGL PyOpenGL_accelerate glfw")

import ccxt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Create output directory
output_dir = 'wavelet_plots_opengl'
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("GPU-ACCELERATED WAVELET DECOMPOSITION - OPENGL MODE")
print("=" * 70)

# =============================================================================
# STEP 1: INITIALIZE OPENGL CONTEXT
# =============================================================================

ctx_initialized = False
use_gpu = False
window = None

def is_raspberry_pi_zero():
    """Detect if running on Raspberry Pi Zero (VideoCore IV - no compute support)."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().lower()
            return 'zero' in model or 'pi 1' in model or 'pi 2' in model or 'pi 3' in model
    except:
        pass
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read().lower()
            # BCM2835 is VideoCore IV (Pi Zero, Pi 1)
            # BCM2836/BCM2837 is also VideoCore IV (Pi 2, Pi 3)
            if 'bcm2835' in cpuinfo or 'bcm2836' in cpuinfo or 'bcm2837' in cpuinfo:
                return True
    except:
        pass
    return False

if OPENGL_AVAILABLE:
    # Skip OpenGL entirely on Pi Zero/older models - they don't support compute shaders
    # and attempting to create contexts can cause bus errors
    if is_raspberry_pi_zero():
        print(f"\n⚠ Raspberry Pi Zero/older model detected (VideoCore IV)")
        print(f"  VideoCore IV only supports OpenGL ES 2.0 - no compute shaders")
        print(f"  Using CPU fallback to avoid bus errors")
        OPENGL_AVAILABLE = False

if OPENGL_AVAILABLE:
    try:
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Request OpenGL 4.3 core profile (for compute shaders)
        # Fall back to OpenGL ES 3.1 on embedded devices (Raspberry Pi)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Headless context
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        # Try to create window with OpenGL 4.3
        window = glfw.create_window(1, 1, "Compute", None, None)
        
        if not window:
            # Fall back to default context (for OpenGL ES on RPi)
            glfw.default_window_hints()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
            window = glfw.create_window(1, 1, "Compute", None, None)
        
        if not window:
            # Last resort: any available context
            glfw.default_window_hints()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            window = glfw.create_window(1, 1, "Compute", None, None)
        
        if window:
            glfw.make_context_current(window)
            
            # Get OpenGL info
            vendor = glGetString(GL_VENDOR)
            renderer = glGetString(GL_RENDERER)
            version = glGetString(GL_VERSION)
            
            if vendor:
                vendor = vendor.decode('utf-8')
            if renderer:
                renderer = renderer.decode('utf-8')
            if version:
                version = version.decode('utf-8')
            
            print(f"\n✓ OpenGL Initialized")
            print(f"  Vendor:   {vendor}")
            print(f"  Renderer: {renderer}")
            print(f"  Version:  {version}")
            
            # Check for compute shader support
            major_version = glGetIntegerv(GL_MAJOR_VERSION)
            minor_version = glGetIntegerv(GL_MINOR_VERSION)
            
            # Compute shaders require OpenGL 4.3+ or OpenGL ES 3.1+
            has_compute = (major_version > 4) or (major_version == 4 and minor_version >= 3)
            
            # Check for compute shader extension as fallback
            if not has_compute:
                extensions = glGetString(GL_EXTENSIONS)
                if extensions:
                    extensions = extensions.decode('utf-8')
                    has_compute = 'GL_ARB_compute_shader' in extensions or 'compute_shader' in extensions.lower()
            
            if has_compute:
                print(f"  Compute Shaders: ✓ Supported")
                
                # Get compute shader limits
                try:
                    max_work_group_count = [
                        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i)[0] for i in range(3)
                    ]
                    max_work_group_size = [
                        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i)[0] for i in range(3)
                    ]
                    max_work_group_invocations = glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)
                    
                    print(f"  Max Work Group Count: {max_work_group_count}")
                    print(f"  Max Work Group Size:  {max_work_group_size}")
                    print(f"  Max Invocations:      {max_work_group_invocations}")
                except:
                    print(f"  (Could not query compute limits)")
                
                ctx_initialized = True
                use_gpu = True
            else:
                print(f"  Compute Shaders: ✗ Not supported (need OpenGL 4.3+ or ES 3.1+)")
                print(f"  Falling back to CPU")
        else:
            print(f"\n⚠ Could not create OpenGL context")
            
    except Exception as e:
        print(f"\n⚠ OpenGL initialization failed: {e}")

if not use_gpu:
    print(f"\n⚠ Using CPU fallback (NumPy convolution)")

print(f"  Output Directory: {output_dir}/\n")

# =============================================================================
# STEP 2: DEFINE COMPUTE SHADER
# =============================================================================

CONVOLUTION_SHADER = """
#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer SignalBuffer {
    float signal[];
};

layout(std430, binding = 1) buffer FilterBuffer {
    float filter_coeffs[];
};

layout(std430, binding = 2) buffer OutputBuffer {
    float output_data[];
};

uniform int signal_length;
uniform int filter_length;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    // Bounds check
    if (i >= signal_length - filter_length + 1) {
        return;
    }
    
    // Convolution: sum of element-wise multiplication
    float sum = 0.0;
    for (int j = 0; j < filter_length; j++) {
        sum += signal[i + j] * filter_coeffs[j];
    }
    
    output_data[i] = sum;
}
"""

# OpenGL ES 3.1 version for Raspberry Pi
CONVOLUTION_SHADER_ES = """
#version 310 es

layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer SignalBuffer {
    highp float signal[];
};

layout(std430, binding = 1) buffer FilterBuffer {
    highp float filter_coeffs[];
};

layout(std430, binding = 2) buffer OutputBuffer {
    highp float output_data[];
};

uniform int signal_length;
uniform int filter_length;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= uint(signal_length - filter_length + 1)) {
        return;
    }
    
    highp float sum = 0.0;
    for (int j = 0; j < filter_length; j++) {
        sum += signal[i + uint(j)] * filter_coeffs[j];
    }
    
    output_data[i] = sum;
}
"""

# Compile shader if OpenGL available
compute_program = None
if use_gpu:
    try:
        # Try desktop OpenGL shader first
        try:
            compute_shader = shaders.compileShader(CONVOLUTION_SHADER, GL_COMPUTE_SHADER)
            compute_program = shaders.compileProgram(compute_shader)
            print("✓ Compute shader compiled (OpenGL 4.3)\n")
        except:
            # Fall back to OpenGL ES shader
            compute_shader = shaders.compileShader(CONVOLUTION_SHADER_ES, GL_COMPUTE_SHADER)
            compute_program = shaders.compileProgram(compute_shader)
            print("✓ Compute shader compiled (OpenGL ES 3.1)\n")
    except Exception as e:
        print(f"✗ Shader compilation failed: {e}")
        print("  Falling back to CPU\n")
        use_gpu = False
        compute_program = None

# =============================================================================
# STEP 3: DEFINE WAVELETS
# =============================================================================

# Normalized wavelets (sum to 1.0 for price preservation)
haar_low_pass = np.array([0.5, 0.5], dtype=np.float32)
haar_high_pass = np.array([0.5, -0.5], dtype=np.float32)

db4_low_pass = np.array([
    0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551
], dtype=np.float32)
db4_sum = db4_low_pass.sum()
db4_low_pass = db4_low_pass / db4_sum

print("✓ Wavelets loaded (Haar and DB4)\n")

# =============================================================================
# STEP 4: GPU CONVOLUTION FUNCTION
# =============================================================================

def gpu_convolve_opengl(signal, filter_coeffs):
    """
    Perform 1D convolution using OpenGL compute shaders.
    
    Args:
        signal: Input signal (numpy float32 array)
        filter_coeffs: Filter coefficients (numpy float32 array)
    
    Returns:
        Convolved output (numpy float32 array)
    """
    if not use_gpu or compute_program is None:
        return cpu_convolve(signal, filter_coeffs)
    
    sig_len = len(signal)
    filt_len = len(filter_coeffs)
    output_len = sig_len - filt_len + 1
    
    # Ensure float32
    signal = np.asarray(signal, dtype=np.float32)
    filter_coeffs = np.asarray(filter_coeffs, dtype=np.float32)
    output = np.zeros(output_len, dtype=np.float32)
    
    try:
        # Create SSBOs (Shader Storage Buffer Objects)
        signal_ssbo = glGenBuffers(1)
        filter_ssbo = glGenBuffers(1)
        output_ssbo = glGenBuffers(1)
        
        # Upload signal data
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, signal_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, signal.nbytes, signal, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, signal_ssbo)
        
        # Upload filter data
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, filter_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, filter_coeffs.nbytes, filter_coeffs, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, filter_ssbo)
        
        # Allocate output buffer
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, output.nbytes, None, GL_DYNAMIC_READ)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, output_ssbo)
        
        # Use compute program
        glUseProgram(compute_program)
        
        # Set uniforms
        glUniform1i(glGetUniformLocation(compute_program, "signal_length"), sig_len)
        glUniform1i(glGetUniformLocation(compute_program, "filter_length"), filt_len)
        
        # Dispatch compute shader
        # Work group size is 256 (or 64 for ES), so we need ceil(output_len / 256) groups
        work_group_size = 256  # Match local_size_x in shader
        num_groups = (output_len + work_group_size - 1) // work_group_size
        glDispatchCompute(num_groups, 1, 1)
        
        # Memory barrier to ensure compute shader completes
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        
        # Read back results
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_ssbo)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, output.nbytes, output)
        
        # Cleanup
        glDeleteBuffers(3, [signal_ssbo, filter_ssbo, output_ssbo])
        
        return output
        
    except Exception as e:
        print(f"  GPU convolution failed: {e}, using CPU fallback")
        return cpu_convolve(signal, filter_coeffs)


def cpu_convolve(signal, filter_coeffs):
    """CPU fallback using NumPy."""
    return np.convolve(signal, filter_coeffs, mode='valid').astype(np.float32)


def convolve(signal, filter_coeffs):
    """Unified convolution function - uses GPU if available, else CPU."""
    if use_gpu:
        return gpu_convolve_opengl(signal, filter_coeffs)
    return cpu_convolve(signal, filter_coeffs)

# =============================================================================
# STEP 5: FETCH REAL BTC DATA
# =============================================================================

print("=" * 70)
print("FETCHING BTC DATA FROM BINANCE")
print("=" * 70)

try:
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = 'BTC/USDT'
    timeframe = '5m'
    
    # Calculate timestamp for 2 weeks back
    two_weeks_ago = datetime.now() - timedelta(weeks=2)
    since = int(two_weeks_ago.timestamp() * 1000)
    
    print(f"\n  Downloading {symbol} {timeframe} data from {two_weeks_ago.strftime('%Y-%m-%d %H:%M')}... ", end='', flush=True)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    print("✓")
    
    timestamps = np.array([candle[0] for candle in ohlcv])
    prices = np.array([candle[4] for candle in ohlcv], dtype=np.float32)
    volumes = np.array([candle[5] for candle in ohlcv], dtype=np.float32)
    dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]
    
    print(f"  Data range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Current BTC: ${prices[-1]:,.2f}")
    print(f"  Change: {((prices[-1] - prices[0]) / prices[0] * 100):+.2f}%")
    print(f"  Data points: {len(prices)}\n")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("  Generating synthetic data...")
    
    # Fallback to synthetic data
    np.random.seed(42)
    n_points = 1000
    initial_price = 50000.0
    drift = 0.0001
    volatility = 0.02
    returns = np.random.randn(n_points) * volatility + drift
    price_multipliers = np.exp(np.cumsum(returns))
    prices = (initial_price * price_multipliers).astype(np.float32)
    volumes = np.random.uniform(100, 1000, n_points).astype(np.float32)
    dates = [datetime.now() - timedelta(minutes=5*(n_points-i)) for i in range(n_points)]
    
    print(f"  Generated {n_points} synthetic price points\n")

# =============================================================================
# STEP 6: PERFORM WAVELET DECOMPOSITION
# =============================================================================

print("=" * 70)
print("COMPUTING WAVELET DECOMPOSITION")
print("=" * 70)

start = time.time()
trend_gpu = convolve(prices, haar_low_pass)
detail_gpu = convolve(prices, haar_high_pass)
trend_db4 = convolve(prices, db4_low_pass)
compute_time = time.time() - start

mode_str = "GPU (OpenGL)" if use_gpu else "CPU (NumPy)"
print(f"\n✓ Decomposition complete ({mode_str}): {compute_time*1000:.2f}ms")
print(f"  Trend points: {len(trend_gpu)}")
print(f"  Detail points: {len(detail_gpu)}\n")

# Multi-level decomposition
approximations = []
details = []
current_signal = prices

for i in range(5):
    approx = convolve(current_signal, haar_low_pass)
    detail = convolve(current_signal, haar_high_pass)
    approximations.append(approx)
    details.append(detail)
    current_signal = approx

levels = approximations

# =============================================================================
# STEP 7: ANOMALY DETECTION
# =============================================================================

detail_abs = np.abs(detail_gpu)
median = np.median(detail_abs)
mad = np.median(np.abs(detail_abs - median))
threshold = 3.0
anomaly_threshold = median + threshold * mad
anomaly_indices = np.where(detail_abs > anomaly_threshold)[0]

print("=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# =============================================================================
# PLOT 1: MAIN OVERVIEW (4 SUBPLOTS)
# =============================================================================

print("\n[1/5] Main overview plot... ", end='', flush=True)

fig = plt.figure(figsize=(16, 12), dpi=300)
gs = GridSpec(4, 1, figure=fig, hspace=0.3)

offset = len(prices) - len(trend_gpu)
aligned_dates = dates[offset:]

# Subplot 1: Original prices
ax1 = fig.add_subplot(gs[0])
ax1.plot(dates, prices, 'b-', linewidth=1, alpha=0.7, label='Original Price')
ax1.set_title('BTC/USDT Price History (5-min candles)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price (USD)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Subplot 2: Price with Trend
ax2 = fig.add_subplot(gs[1])
ax2.plot(dates, prices, 'b-', linewidth=1, alpha=0.4, label='Original Price')
ax2.plot(aligned_dates, trend_gpu, 'r-', linewidth=2, label='Trend (Low-pass)')
ax2.set_title('Price Decomposition: Original vs Trend', fontsize=14, fontweight='bold')
ax2.set_ylabel('Price (USD)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Subplot 3: Detail coefficients
ax3 = fig.add_subplot(gs[2])
ax3.plot(aligned_dates, detail_gpu, 'g-', linewidth=1, alpha=0.7, label='Detail (High-pass)')
ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax3.axhline(y=anomaly_threshold, color='r', linestyle='--', linewidth=1, label=f'Anomaly Threshold (${anomaly_threshold:.0f})')
ax3.axhline(y=-anomaly_threshold, color='r', linestyle='--', linewidth=1)
if len(anomaly_indices) > 0:
    anomaly_dates = [aligned_dates[i] for i in anomaly_indices if i < len(aligned_dates)]
    anomaly_values = detail_gpu[anomaly_indices[:len(anomaly_dates)]]
    ax3.scatter(anomaly_dates, anomaly_values, color='red', s=50, marker='*', 
               zorder=5, label=f'Anomalies ({len(anomaly_indices)})')
ax3.set_title('Detail Coefficients (High-Frequency Changes)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Detail Coefficient (USD)', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Subplot 4: Volume
ax4 = fig.add_subplot(gs[3])
ax4.bar(dates, volumes, width=0.04, color='purple', alpha=0.6)
ax4.set_title('Trading Volume', fontsize=14, fontweight='bold')
ax4.set_ylabel('Volume (BTC)', fontsize=11)
ax4.set_xlabel('Date', fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.tight_layout()
plt.savefig(f'{output_dir}/01_main_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 2: PROGRESSIVE APPROXIMATIONS
# =============================================================================

print("[2/5] Progressive approximations... ", end='', flush=True)

fig = plt.figure(figsize=(20, 16), dpi=300)
gs = GridSpec(6, 1, figure=fig, hspace=0.35)
fig.suptitle('Progressive Approximations - Frequency Filtering (OpenGL Compute)', fontsize=18, fontweight='bold')

ax_orig = plt.subplot(gs[0])
ax_orig.plot(dates, prices, 'b-', linewidth=1.5, alpha=0.8, label='Original')
ax_orig.set_title('Original Signal: BTC/USDT', fontsize=12, fontweight='bold')
ax_orig.set_ylabel('Price (USD)', fontsize=10)
ax_orig.grid(True, alpha=0.3)
ax_orig.legend(loc='upper left')
ax_orig.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

colors = ['orangered', 'orange', 'gold', 'limegreen', 'dodgerblue']
labels = ['Level 1 (10min)', 'Level 2 (20min)', 'Level 3 (40min)', 'Level 4 (80min)', 'Level 5 (160min)']

for i in range(5):
    ax = plt.subplot(gs[i+1])
    offset_a = len(prices) - len(approximations[i])
    dates_a = dates[offset_a:]
    ax.plot(dates_a, approximations[i], linewidth=2, color=colors[i], label=labels[i])
    ax.set_title(f'Approximation {labels[i]}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax.set_xlabel('Date', fontsize=10)
plt.savefig(f'{output_dir}/02_progressive_approximations.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 3: FREQUENCY BANDS (DETAILS)
# =============================================================================

print("[3/5] Frequency bands... ", end='', flush=True)

fig = plt.figure(figsize=(20, 16), dpi=300)
gs = GridSpec(6, 1, figure=fig, hspace=0.35)
fig.suptitle('Frequency Band Decomposition - Detail Coefficients', fontsize=18, fontweight='bold')

ax_orig = plt.subplot(gs[0])
ax_orig.plot(dates, prices, 'b-', linewidth=1.5, alpha=0.8)
ax_orig.set_title('Original Signal', fontsize=12, fontweight='bold')
ax_orig.set_ylabel('Price (USD)', fontsize=10)
ax_orig.grid(True, alpha=0.3)
ax_orig.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

detail_colors = ['red', 'orange', 'gold', 'green', 'blue']

for i in range(5):
    ax = plt.subplot(gs[i+1])
    offset_d = len(prices) - len(details[i])
    dates_d = dates[offset_d:]
    ax.plot(dates_d, details[i], linewidth=1, color=detail_colors[i], alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.fill_between(dates_d, 0, details[i], alpha=0.3, color=detail_colors[i])
    ax.set_title(f'Detail Band {i+1}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coefficient', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax.set_xlabel('Date', fontsize=10)
plt.savefig(f'{output_dir}/03_frequency_bands.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 4: ANOMALY DETECTION
# =============================================================================

print("[4/5] Anomaly detection... ", end='', flush=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=300)
fig.suptitle('Anomaly Detection Analysis (OpenGL Compute)', fontsize=16, fontweight='bold')

ax1.plot(aligned_dates, detail_gpu, 'b-', linewidth=1, alpha=0.7, label='Detail Coefficients')
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax1.axhline(y=anomaly_threshold, color='r', linestyle='--', linewidth=1.5, 
           label=f'Threshold: ${anomaly_threshold:.2f}')
ax1.axhline(y=-anomaly_threshold, color='r', linestyle='--', linewidth=1.5)

if len(anomaly_indices) > 0:
    anomaly_dates = [aligned_dates[i] for i in anomaly_indices if i < len(aligned_dates)]
    anomaly_values = detail_gpu[anomaly_indices[:len(anomaly_dates)]]
    ax1.scatter(anomaly_dates, anomaly_values, color='red', s=100, marker='*', 
               zorder=5, edgecolors='darkred', linewidth=1, label=f'Anomalies: {len(anomaly_indices)}')

ax1.set_title('Detail Coefficients with Anomaly Markers', fontsize=13)
ax1.set_ylabel('Detail Coefficient (USD)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=10)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax2.plot(aligned_dates, detail_abs, 'purple', linewidth=1, alpha=0.7, label='Absolute Detail')
ax2.axhline(y=anomaly_threshold, color='r', linestyle='--', linewidth=1.5)
ax2.fill_between(aligned_dates, 0, anomaly_threshold, alpha=0.2, color='green', label='Normal Range')
ax2.fill_between(aligned_dates, anomaly_threshold, detail_abs.max(), alpha=0.2, color='red', label='Anomaly Zone')

ax2.set_title('Volatility Measure (Absolute Detail)', fontsize=13)
ax2.set_ylabel('Absolute Detail (USD)', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.subplots_adjust(hspace=0.3)
plt.savefig(f'{output_dir}/04_anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# PLOT 5: TRADING SIGNALS
# =============================================================================

print("[5/5] Trading signals... ", end='', flush=True)

trend = levels[2]
offset = len(prices) - len(trend)
aligned_prices = prices[offset:]
signal_dates = dates[offset:]
price_deviation = aligned_prices - trend
buffer = np.std(price_deviation) * 0.5

buy_signals = price_deviation < -buffer
sell_signals = price_deviation > buffer

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=300)
fig.suptitle('Trading Signal Generation (OpenGL Compute)', fontsize=16, fontweight='bold')

ax1.plot(signal_dates, aligned_prices, 'b-', linewidth=1.5, alpha=0.7, label='Price')
ax1.plot(signal_dates, trend, 'orange', linewidth=2, label='Trend (Level 3)')
ax1.fill_between(signal_dates, trend - buffer, trend + buffer, alpha=0.2, color='gray', label='Neutral Zone')

buy_dates = [signal_dates[i] for i in range(len(buy_signals)) if buy_signals[i]]
buy_prices = aligned_prices[buy_signals]
sell_dates = [signal_dates[i] for i in range(len(sell_signals)) if sell_signals[i]]
sell_prices = aligned_prices[sell_signals]

ax1.scatter(buy_dates, buy_prices, color='green', s=100, marker='^', 
           zorder=5, label=f'BUY ({buy_signals.sum()})', edgecolors='darkgreen', linewidth=1)
ax1.scatter(sell_dates, sell_prices, color='red', s=100, marker='v', 
           zorder=5, label=f'SELL ({sell_signals.sum()})', edgecolors='darkred', linewidth=1)

ax1.set_title('Price with Trading Signals', fontsize=13)
ax1.set_ylabel('Price (USD)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=10)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax2.plot(signal_dates, price_deviation, 'purple', linewidth=1.5, alpha=0.7, label='Price Deviation')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.axhline(y=buffer, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Sell Threshold')
ax2.axhline(y=-buffer, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Buy Threshold')
ax2.fill_between(signal_dates, -buffer, buffer, alpha=0.2, color='gray')

ax2.set_title('Deviation from Trend', fontsize=13)
ax2.set_ylabel('Deviation (USD)', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.subplots_adjust(hspace=0.3)
plt.savefig(f'{output_dir}/05_trading_signals.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# =============================================================================
# CLEANUP
# =============================================================================

if OPENGL_AVAILABLE and ctx_initialized:
    try:
        # Ensure all GL operations complete before cleanup
        if use_gpu:
            try:
                glFinish()  # Wait for all GL commands to complete
            except:
                pass
        if window:
            glfw.destroy_window(window)
        glfw.terminate()
    except Exception as e:
        # Silently handle cleanup errors to avoid bus errors on exit
        pass

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\n✓ All plots saved to '{output_dir}/' directory")
print(f"\nGenerated files:")
print(f"  1. 01_main_overview.png          - 4-panel overview")
print(f"  2. 02_progressive_approximations.png - Multi-level filtering")
print(f"  3. 03_frequency_bands.png        - Detail coefficients")
print(f"  4. 04_anomaly_detection.png      - Anomaly analysis")
print(f"  5. 05_trading_signals.png        - Buy/sell signals")
print(f"\nProcessing Mode: {'GPU (OpenGL Compute)' if use_gpu else 'CPU (NumPy fallback)'}")
print(f"Processing Time: {compute_time*1000:.2f}ms")
if use_gpu:
    print(f"GPU: {renderer if 'renderer' in dir() else 'Unknown'}")
print("=" * 70)
