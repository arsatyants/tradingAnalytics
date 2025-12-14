#!/usr/bin/env python3
"""
Simple Web Server for Wavelet Analysis Visualization
====================================================

Provides a web interface to:
1. Select cryptocurrency (BTC, ETH, SOL)
2. Run GPU wavelet analysis
3. Display generated plots in the browser

Usage:
    python web_server.py
    
Then open: http://localhost:8080
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess
import json
import os
import time
from urllib.parse import urlparse, parse_qs
import mimetypes

# Configuration
PORT = 8080
PLOT_SCRIPT = 'gpu_wavelet_gpu_plot.py'
PYTHON_BIN = '.venv/bin/python'
OUTPUT_DIR = 'wavelet_plots'

class WaveletHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.serve_index()
        elif path == '/api/currencies':
            self.serve_currencies()
        elif path.startswith('/plots/'):
            self.serve_image(path)
        elif path == '/favicon.ico':
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'404 Not Found')
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/analyze':
            self.handle_analyze()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'404 Not Found')
    
    def serve_index(self):
        """Serve the main HTML page"""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Wavelet Analysis - Trading Analytics</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .controls {
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .currency-btn {
            padding: 15px 40px;
            font-size: 1.1em;
            font-weight: bold;
            border: 3px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .currency-btn:hover {
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .currency-btn.active {
            background: #667eea;
            color: white;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .timeframe-select {
            padding: 12px 25px;
            font-size: 1em;
            font-weight: 600;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .timeframe-select:hover {
            background: #f0f3ff;
        }
        
        .timeframe-label {
            font-weight: bold;
            color: #667eea;
            margin-right: 10px;
        }
        
        .analyze-btn {
            padding: 15px 60px;
            font-size: 1.2em;
            font-weight: bold;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .analyze-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(245, 87, 108, 0.4);
        }
        
        .analyze-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .status {
            padding: 20px 30px;
            text-align: center;
            font-size: 1.1em;
            min-height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .status.loading {
            color: #667eea;
            font-weight: bold;
        }
        
        .status.success {
            color: #28a745;
            font-weight: bold;
        }
        
        .status.error {
            color: #dc3545;
            font-weight: bold;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(102, 126, 234, 0.3);
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .plots-container {
            padding: 30px;
        }
        
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
        }
        
        .plot-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .plot-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }
        
        .plot-card img {
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }
        
        .plot-title {
            padding: 15px;
            background: #f8f9fa;
            font-weight: bold;
            text-align: center;
            color: #495057;
            border-top: 3px solid #667eea;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 30px;
            color: #6c757d;
        }
        
        .empty-state h2 {
            font-size: 1.8em;
            margin-bottom: 15px;
            color: #495057;
        }
        
        .empty-state p {
            font-size: 1.1em;
            line-height: 1.6;
        }
        
        .metrics-panel {
            padding: 20px 30px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            border-top: 3px solid #667eea;
            border-bottom: 3px solid #667eea;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-value.positive {
            color: #28a745;
        }
        
        .metric-value.negative {
            color: #dc3545;
        }
        
        .frequency-bands {
            padding: 30px;
            background: #f8f9fa;
        }
        
        .frequency-bands h2 {
            text-align: center;
            color: #495057;
            margin-bottom: 25px;
            font-size: 1.8em;
        }
        
        .bands-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        
        .band-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        
        .band-card h3 {
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        
        .band-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            font-size: 0.95em;
        }
        
        .band-info-item {
            display: flex;
            flex-direction: column;
        }
        
        .band-info-label {
            color: #6c757d;
            font-size: 0.85em;
            margin-bottom: 4px;
        }
        
        .band-info-value {
            color: #495057;
            font-weight: bold;
            font-size: 1.1em;
        }
        
        footer {
            padding: 20px;
            text-align: center;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }
        
        /* Modal for full-size images */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            align-items: center;
            justify-content: center;
        }
        
        .modal.active {
            display: flex;
        }
        
        .modal img {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        }
        
        .modal-close {
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s;
        }
        
        .modal-close:hover {
            color: #f5576c;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 GPU Wavelet Analysis</h1>
            <p class="subtitle">Cryptocurrency Trading Analytics with 8-Level Decomposition</p>
        </header>
        
        <div class="controls">
            <div class="control-group">
                <button class="currency-btn active" data-currency="BTC">
                    Bitcoin (BTC)
                </button>
                <button class="currency-btn" data-currency="ETH">
                    Ethereum (ETH)
                </button>
                <button class="currency-btn" data-currency="SOL">
                    Solana (SOL)
                </button>
            </div>
            <div class="control-group" style="margin-top: 20px;">
                <span class="timeframe-label">📊 Timeframe:</span>
                <select id="timeframe-select" class="timeframe-select">
                    <option value="1m">1 Minute</option>
                    <option value="5m" selected>5 Minutes</option>
                    <option value="15m">15 Minutes</option>
                    <option value="30m">30 Minutes</option>
                    <option value="1h">1 Hour</option>
                    <option value="4h">4 Hours</option>
                    <option value="1d">1 Day</option>
                </select>
                <button class="analyze-btn" onclick="runAnalysis()">
                    ⚡ Run Analysis
                </button>
            </div>
        </div>
        
        <div class="status" id="status"></div>
        
        <div class="metrics-panel" id="metrics-panel" style="display: none;"></div>
        
        <div class="frequency-bands" id="frequency-bands" style="display: none;">
            <h2>📊 Frequency Band Analysis (8 Levels)</h2>
            <p id="frequency-bands-info" style="text-align: center; color: #6c757d; margin-bottom: 20px;">
                Based on <strong id="timeframe-display">5-minute</strong> candle data (1000 samples)
            </p>
            <div class="bands-grid">
                <div class="band-card">
                    <h3>🔹 Band 1: Very High Frequency (10-20min)</h3>
                    <div class="band-info">
                        <div class="band-info-item">
                            <span class="band-info-label">Period Range</span>
                            <span class="band-info-value">10-20 minutes</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Typical Use</span>
                            <span class="band-info-value">Scalping</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Frequency Type</span>
                            <span class="band-info-value">Sub-hourly</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Downsampling</span>
                            <span class="band-info-value">2× (10min)</span>
                        </div>
                    </div>
                </div>
                
                <div class="band-card">
                    <h3>🔹 Band 2: High Frequency (20-40min)</h3>
                    <div class="band-info">
                        <div class="band-info-item">
                            <span class="band-info-label">Period Range</span>
                            <span class="band-info-value">20-40 minutes</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Typical Use</span>
                            <span class="band-info-value">Quick trades</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Frequency Type</span>
                            <span class="band-info-value">Hourly</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Downsampling</span>
                            <span class="band-info-value">4× (20min)</span>
                        </div>
                    </div>
                </div>
                
                <div class="band-card">
                    <h3>🔹 Band 3: Medium-High (40min-1.3h)</h3>
                    <div class="band-info">
                        <div class="band-info-item">
                            <span class="band-info-label">Period Range</span>
                            <span class="band-info-value">40-80 minutes</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Typical Use</span>
                            <span class="band-info-value">Intraday swings</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Frequency Type</span>
                            <span class="band-info-value">1-2 Hours</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Downsampling</span>
                            <span class="band-info-value">8× (40min)</span>
                        </div>
                    </div>
                </div>
                
                <div class="band-card">
                    <h3>🔹 Band 4: Medium (1.3-2.7h)</h3>
                    <div class="band-info">
                        <div class="band-info-item">
                            <span class="band-info-label">Period Range</span>
                            <span class="band-info-value">1.3-2.7 hours</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Typical Use</span>
                            <span class="band-info-value">Session trends</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Frequency Type</span>
                            <span class="band-info-value">Few Hours</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Downsampling</span>
                            <span class="band-info-value">16× (1.3h)</span>
                        </div>
                    </div>
                </div>
                
                <div class="band-card">
                    <h3>🔹 Band 5: Medium-Low (2.7-5.3h)</h3>
                    <div class="band-info">
                        <div class="band-info-item">
                            <span class="band-info-label">Period Range</span>
                            <span class="band-info-value">2.7-5.3 hours</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Typical Use</span>
                            <span class="band-info-value">Half-day patterns</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Frequency Type</span>
                            <span class="band-info-value">Quarter Day</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Downsampling</span>
                            <span class="band-info-value">32× (2.7h)</span>
                        </div>
                    </div>
                </div>
                
                <div class="band-card">
                    <h3>🔹 Band 6: Low Frequency (5.3-10.7h)</h3>
                    <div class="band-info">
                        <div class="band-info-item">
                            <span class="band-info-label">Period Range</span>
                            <span class="band-info-value">5.3-10.7 hours</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Typical Use</span>
                            <span class="band-info-value">Daily cycles</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Frequency Type</span>
                            <span class="band-info-value">Half Day</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Downsampling</span>
                            <span class="band-info-value">64× (5.3h)</span>
                        </div>
                    </div>
                </div>
                
                <div class="band-card">
                    <h3>🔹 Band 7: Very Low (10.7-21.3h)</h3>
                    <div class="band-info">
                        <div class="band-info-item">
                            <span class="band-info-label">Period Range</span>
                            <span class="band-info-value">10.7-21.3 hours</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Typical Use</span>
                            <span class="band-info-value">Day trading</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Frequency Type</span>
                            <span class="band-info-value">1-2 Days</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Downsampling</span>
                            <span class="band-info-value">128× (10.7h)</span>
                        </div>
                    </div>
                </div>
                
                <div class="band-card">
                    <h3>🔹 Band 8: Ultra Low (21.3h+)</h3>
                    <div class="band-info">
                        <div class="band-info-item">
                            <span class="band-info-label">Period Range</span>
                            <span class="band-info-value">21+ hours</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Typical Use</span>
                            <span class="band-info-value">Swing trading</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Frequency Type</span>
                            <span class="band-info-value">Multi-day Trend</span>
                        </div>
                        <div class="band-info-item">
                            <span class="band-info-label">Downsampling</span>
                            <span class="band-info-value">256× (21.3h)</span>
                        </div>
                    </div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 25px; padding: 20px; background: white; border-radius: 12px;">
                <p style="color: #6c757d; font-size: 0.95em; line-height: 1.6;">
                    <strong>Note:</strong> Time periods are calculated based on <strong>5-minute candles</strong> with dyadic downsampling (2ⁿ). 
                    Each decomposition level doubles the effective timeframe. The analysis uses 1000 samples covering approximately 
                    <strong>3.5 days</strong> of trading data. Actual oscillation periods and amplitudes vary by cryptocurrency 
                    and market conditions. These frequency bands help identify price patterns at different time scales for 
                    developing trading strategies (scalping, day trading, swing trading).
                </p>
            </div>
        </div>
        
        <div class="plots-container" id="plots-container">
            <div class="empty-state">
                <h2>📊 Ready to Analyze</h2>
                <p>Select a cryptocurrency and click "Run Analysis" to generate wavelet decomposition plots.</p>
                <p style="margin-top: 10px;">The analysis includes:</p>
                <ul style="list-style: none; margin-top: 10px; line-height: 2;">
                    <li>📈 Price overview with trend extraction</li>
                    <li>🔬 8-level frequency decomposition</li>
                    <li>⚠️ Anomaly detection</li>
                    <li>💹 Trading signals</li>
                    <li>📊 Statistical dashboard</li>
                </ul>
            </div>
        </div>
        
        <footer>
            GPU-Accelerated Wavelet Analysis | NVIDIA GeForce RTX 4060 | ~0.6ms processing time
        </footer>
    </div>
    
    <!-- Modal for full-size images -->
    <div class="modal" id="modal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img id="modal-img" src="" alt="Full size plot">
    </div>
    
    <script>
        let selectedCurrency = 'BTC';
        
        // Currency selection
        document.querySelectorAll('.currency-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.currency-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                selectedCurrency = this.dataset.currency;
            });
        });
        
        // Run analysis
        async function runAnalysis() {
            const statusEl = document.getElementById('status');
            const analyzeBtn = document.querySelector('.analyze-btn');
            const plotsContainer = document.getElementById('plots-container');
            const selectedTimeframe = document.getElementById('timeframe-select').value;
            
            // Disable button and show loading
            analyzeBtn.disabled = true;
            statusEl.className = 'status loading';
            statusEl.innerHTML = '<span class="spinner"></span>Running GPU wavelet analysis for ' + selectedCurrency + '/USDT (' + selectedTimeframe + ')...';
            plotsContainer.innerHTML = '<div class="empty-state"><h2>⚙️ Processing...</h2><p>GPU decomposition in progress</p></div>';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        currency: selectedCurrency,
                        timeframe: selectedTimeframe 
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    statusEl.className = 'status success';
                    statusEl.innerHTML = '✓ Analysis complete! Generated ' + result.plots.length + ' plots in ' + result.execution_time + ' seconds';
                    console.log('Received metrics:', result.metrics);
                    console.log('Frequency bands:', result.metrics.frequency_bands);
                    displayMetrics(result.metrics, selectedCurrency);
                    displayPlots(result.plots, selectedCurrency);
                    displayFrequencyBands(result.metrics.frequency_bands || [], selectedTimeframe);
                } else {
                    throw new Error(result.error || 'Analysis failed');
                }
            } catch (error) {
                statusEl.className = 'status error';
                statusEl.innerHTML = '✗ Error: ' + error.message;
                plotsContainer.innerHTML = '<div class="empty-state"><h2>❌ Error</h2><p>' + error.message + '</p></div>';
            } finally {
                analyzeBtn.disabled = false;
            }
        }
        
        // Display metrics
        function displayMetrics(metrics, currency) {
            const metricsPanel = document.getElementById('metrics-panel');
            
            if (!metrics.current_price && !metrics.price_change && !metrics.gpu_time) {
                metricsPanel.style.display = 'none';
                return;
            }
            
            let html = '';
            
            if (metrics.current_price) {
                html += `
                    <div class="metric-card">
                        <div class="metric-label">Current Price</div>
                        <div class="metric-value">$${parseFloat(metrics.current_price).toLocaleString()}</div>
                    </div>
                `;
            }
            
            if (metrics.price_change) {
                const isPositive = metrics.price_change.includes('+');
                const changeClass = isPositive ? 'positive' : 'negative';
                html += `
                    <div class="metric-card">
                        <div class="metric-label">Price Change</div>
                        <div class="metric-value ${changeClass}">${metrics.price_change}</div>
                    </div>
                `;
            }
            
            if (metrics.gpu_time) {
                html += `
                    <div class="metric-card">
                        <div class="metric-label">GPU Processing</div>
                        <div class="metric-value">${metrics.gpu_time}</div>
                    </div>
                `;
            }
            
            html += `
                <div class="metric-card">
                    <div class="metric-label">Decomposition Levels</div>
                    <div class="metric-value">8 Bands</div>
                </div>
            `;
            
            metricsPanel.innerHTML = html;
            metricsPanel.style.display = 'grid';
        }
        
        // Display frequency bands
        function displayFrequencyBands(bands, timeframe) {
            const container = document.getElementById('frequency-bands');
            
            if (!bands || bands.length === 0) {
                container.style.display = 'none';
                return;
            }
            
            // Update timeframe display
            const timeframeNames = {
                '1m': '1-minute',
                '5m': '5-minute',
                '15m': '15-minute',
                '30m': '30-minute',
                '1h': '1-hour',
                '4h': '4-hour',
                '1d': '1-day'
            };
            document.getElementById('timeframe-display').textContent = timeframeNames[timeframe] || timeframe;
            
            const bandsGrid = container.querySelector('.bands-grid');
            const bandDescriptions = [
                { name: 'Very High Frequency', type: 'Scalping', desc: 'Sub-hourly' },
                { name: 'High Frequency', type: 'Quick trades', desc: 'Hourly' },
                { name: 'Medium-High', type: 'Intraday swings', desc: '1-2 Hours' },
                { name: 'Medium', type: 'Session trends', desc: 'Few Hours' },
                { name: 'Medium-Low', type: 'Half-day patterns', desc: 'Quarter Day' },
                { name: 'Low Frequency', type: 'Daily cycles', desc: 'Half Day' },
                { name: 'Very Low', type: 'Day trading', desc: '1-2 Days' },
                { name: 'Ultra Low', type: 'Swing trading', desc: 'Multi-day Trend' }
            ];
            
            function formatPeriod(hours) {
                if (hours === 0) return 'N/A';
                if (hours < 1) return (hours * 60).toFixed(0) + ' min';
                if (hours < 24) return hours.toFixed(1) + ' h';
                return (hours / 24).toFixed(1) + ' days';
            }
            
            let html = '';
            bands.forEach((band, i) => {
                const desc = bandDescriptions[i] || bandDescriptions[7];
                const downsample = Math.pow(2, i + 1);
                
                // Check if we have actual measured data
                const hasData = band.min_to_min_hours !== undefined;
                
                html += `
                    <div class="band-card">
                        <h3>🔹 Band ${band.level}: ${desc.name}</h3>
                        <div class="band-info">
                            <div class="band-info-item">
                                <span class="band-info-label">Min→Min Period</span>
                                <span class="band-info-value">${hasData ? formatPeriod(band.min_to_min_hours) : 'Calculating...'}</span>
                            </div>
                            <div class="band-info-item">
                                <span class="band-info-label">Max→Max Period</span>
                                <span class="band-info-value">${hasData ? formatPeriod(band.max_to_max_hours) : 'Calculating...'}</span>
                            </div>
                            <div class="band-info-item">
                                <span class="band-info-label">Avg Amplitude</span>
                                <span class="band-info-value">${hasData ? '$' + band.amplitude.toFixed(2) : 'Calculating...'}</span>
                            </div>
                            <div class="band-info-item">
                                <span class="band-info-label">Trading Style</span>
                                <span class="band-info-value">${desc.type}</span>
                            </div>
                        </div>
                        <div style="margin-top: 10px; padding: 8px; background: #f8f9fa; border-radius: 6px; font-size: 0.85em; color: #495057;">
                            <strong>Timeframe:</strong> ${desc.desc} | <strong>Downsampling:</strong> ${downsample}×
                        </div>
                    </div>
                `;
            });
            
            bandsGrid.innerHTML = html;
            container.style.display = 'block';
        }
        
        // Display plots
        function displayPlots(plots, currency) {
            const plotsContainer = document.getElementById('plots-container');
            const plotTitles = {
                '01_main_overview.png': '📈 Main Overview - 4 Panel Analysis',
                '02a_progressive_approximations.png': '🔄 Progressive Approximations (8 Levels)',
                '02b_frequency_bands.png': '📊 Frequency Bands (Detail Coefficients)',
                '03_anomaly_detection.png': '⚠️ Anomaly Detection',
                '04_trading_signals.png': '💹 Trading Signals (Buy/Sell)',
                '05_statistics_dashboard.png': '📊 Statistical Dashboard'
            };
            
            let html = '<div class="plot-grid">';
            plots.forEach(plot => {
                const filename = plot.split('/').pop();
                const title = plotTitles[filename] || filename;
                html += `
                    <div class="plot-card">
                        <img src="${plot}?t=${Date.now()}" alt="${title}" onclick="openModal(this.src)">
                        <div class="plot-title">${title}</div>
                    </div>
                `;
            });
            html += '</div>';
            
            plotsContainer.innerHTML = html;
        }
        
        // Modal functions
        function openModal(src) {
            document.getElementById('modal').classList.add('active');
            document.getElementById('modal-img').src = src;
        }
        
        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }
        
        // Close modal on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_currencies(self):
        """Serve list of available currencies"""
        currencies = ['BTC', 'ETH', 'SOL']
        self.send_json_response({'currencies': currencies})
    
    def handle_analyze(self):
        """Handle analysis request"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())
            
            currency = data.get('currency', 'BTC').upper()
            timeframe = data.get('timeframe', '5m').lower()
            
            # Validate currency
            if currency not in ['BTC', 'ETH', 'SOL']:
                self.send_json_response({
                    'success': False,
                    'error': f'Invalid currency: {currency}'
                }, status=400)
                return
            
            # Validate timeframe
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if timeframe not in valid_timeframes:
                self.send_json_response({
                    'success': False,
                    'error': f'Invalid timeframe: {timeframe}'
                }, status=400)
                return
            
            # Run analysis script
            start_time = time.time()
            result = subprocess.run(
                [PYTHON_BIN, PLOT_SCRIPT, currency, timeframe],
                capture_output=True,
                text=True,
                timeout=60
            )
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                self.send_json_response({
                    'success': False,
                    'error': f'Script execution failed: {result.stderr}'
                }, status=500)
                return
            
            # Get list of generated plots
            plot_dir = os.path.join(OUTPUT_DIR, currency.lower())
            if not os.path.exists(plot_dir):
                self.send_json_response({
                    'success': False,
                    'error': 'Plot directory not found'
                }, status=500)
                return
            
            plots = []
            plot_files = [
                '01_main_overview.png',
                '02a_progressive_approximations.png',
                '02b_frequency_bands.png',
                '03_anomaly_detection.png',
                '04_trading_signals.png',
                '05_statistics_dashboard.png'
            ]
            
            for plot_file in plot_files:
                plot_path = os.path.join(plot_dir, plot_file)
                if os.path.exists(plot_path):
                    plots.append(f'/plots/{currency.lower()}/{plot_file}')
            
            # Extract metrics from script output
            metrics = self.extract_metrics(result.stdout)
            
            # Read frequency band metrics from JSON file
            metrics_file = os.path.join(plot_dir, 'metrics.json')
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        freq_data = json.load(f)
                        metrics['frequency_bands'] = freq_data.get('frequency_bands', [])
                except:
                    pass
            
            self.send_json_response({
                'success': True,
                'currency': currency,
                'plots': plots,
                'execution_time': f'{execution_time:.2f}',
                'metrics': metrics
            })
            
        except json.JSONDecodeError:
            self.send_json_response({
                'success': False,
                'error': 'Invalid JSON'
            }, status=400)
        except subprocess.TimeoutExpired:
            self.send_json_response({
                'success': False,
                'error': 'Analysis timeout (>60s)'
            }, status=500)
        except Exception as e:
            self.send_json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def serve_image(self, path):
        """Serve plot images"""
        # Remove leading /plots/ and construct file path
        relative_path = path[7:]  # Remove '/plots/'
        file_path = os.path.join(OUTPUT_DIR, relative_path)
        
        if not os.path.exists(file_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Image not found')
            return
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = 'application/octet-stream'
        
        # Send image
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        
        with open(file_path, 'rb') as f:
            self.wfile.write(f.read())
    
    def extract_metrics(self, script_output):
        """Extract price, GPU metrics, and frequency band data from script output"""
        metrics = {
            'current_price': None,
            'price_change': None,
            'gpu_time': None,
            'frequency_bands': []
        }
        
        lines = script_output.split('\n')
        for line in lines:
            # Extract current price and change
            if 'Current' in line and '$' in line:
                if '$' in line:
                    try:
                        price_part = line.split('$')[1].split()[0]
                        metrics['current_price'] = price_part.replace(',', '')
                    except:
                        pass
            
            if 'Change:' in line and '%' in line:
                try:
                    change_part = line.split('Change:')[1].strip()
                    metrics['price_change'] = change_part
                except:
                    pass
            
            # Extract GPU processing time
            if 'GPU Processing Time:' in line:
                try:
                    time_part = line.split('GPU Processing Time:')[1].strip()
                    metrics['gpu_time'] = time_part
                except:
                    pass
        
        # Generate default frequency bands for 5-minute timeframe
        # These are the theoretical periods based on downsampling
        timeframe_minutes = 5
        for i in range(8):
            level_samples = 2 ** (i + 1)
            min_period_minutes = level_samples * timeframe_minutes
            max_period_minutes = level_samples * 2 * timeframe_minutes
            
            # Format periods
            def format_period(minutes):
                if minutes < 60:
                    return f"{int(minutes)}min"
                else:
                    hours = minutes / 60
                    return f"{hours:.1f}h"
            
            band_data = {
                'level': i + 1,
                'min_period': format_period(min_period_minutes),
                'max_period': format_period(max_period_minutes),
                'min_period_raw': min_period_minutes,
                'max_period_raw': max_period_minutes
            }
            metrics['frequency_bands'].append(band_data)
        
        return metrics
    
    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        """Override to add timestamp to logs"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {format % args}")


def main():
    """Start the web server"""
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, WaveletHandler)
    
    print("=" * 70)
    print("GPU WAVELET ANALYSIS WEB SERVER")
    print("=" * 70)
    print(f"\n✓ Server started on http://localhost:{PORT}")
    print(f"✓ Python binary: {PYTHON_BIN}")
    print(f"✓ Analysis script: {PLOT_SCRIPT}")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"\nOpen your browser and navigate to: http://localhost:{PORT}")
    print("\nPress Ctrl+C to stop the server\n")
    print("=" * 70)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Server stopped")
        print("=" * 70)
        httpd.shutdown()


if __name__ == '__main__':
    main()
