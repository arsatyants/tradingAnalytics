#!/usr/bin/env python3
"""
Parallel Plot Generation Web Server
Runs data preparation once, then generates all 6 plots in parallel using separate processes.
"""
import os
import sys
import json
import time
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse
import uuid

# Configuration
PORT = 8080
# Detect Python binary - prefer .venv if exists
if os.path.exists('.venv/bin/python3'):
    PYTHON_BIN = os.path.abspath('.venv/bin/python3')
elif os.path.exists('.venv/bin/python'):
    PYTHON_BIN = os.path.abspath('.venv/bin/python')
else:
    PYTHON_BIN = sys.executable
OUTPUT_DIR = 'wavelet_plots'
CACHE_MAX_AGE = 10  # seconds

# Job tracking for async execution
jobs = {}
job_lock = threading.Lock()

# Plot script mapping
PLOT_SCRIPTS = [
    'plot_01.py',  # Main overview
    'plot_template.py 02a',  # Progressive approximations
    'plot_template.py 02b',  # Frequency bands
    'plot_template.py 03',   # Anomaly detection
    'plot_template.py 04',   # Trading signals
    'plot_template.py 05',   # Statistics dashboard
]

class ParallelWaveletHandler(BaseHTTPRequestHandler):
    """Handler for parallel plot generation requests"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path.startswith('/api/job/'):
            job_id = path.split('/')[-1]
            self.serve_job_status(job_id)
        elif path == '/' or path == '/index.html':
            self.serve_index()
        elif path == '/api/currencies':
            self.serve_currencies()
        elif path.startswith('/plots/'):
            self.serve_image(path)
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/analyze':
            self.handle_analyze()
        else:
            self.send_error(404)
    
    def serve_index(self):
        """Serve main HTML page"""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Parallel Wavelet Analysis</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        .controls { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }
        .analyze-btn { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .analyze-btn:disabled { background: #ccc; cursor: not-allowed; }
        .status { padding: 15px; margin: 15px 0; border-radius: 5px; font-weight: bold; }
        .status.loading { background: #fff3cd; color: #856404; }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #4CAF50; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        #plots-container { margin-top: 20px; }
        .plot-img { width: 100%; margin: 15px 0; border-radius: 5px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Parallel GPU Wavelet Analysis</h1>
        <div class="controls">
            <label>Currency: <select id="currency-select"><option>BTC</option><option>ETH</option><option>SOL</option></select></label>
            <label>Timeframe: <select id="timeframe-select"><option>5m</option><option>15m</option><option>1h</option></select></label>
            <button class="analyze-btn" onclick="runAnalysis()">Run Parallel Analysis</button>
        </div>
        <div id="status" class="status" style="display:none;"></div>
        <div id="plots-container"></div>
    </div>
    
    <script>
        async function runAnalysis() {
            const statusEl = document.getElementById('status');
            const btn = document.querySelector('.analyze-btn');
            const plotsContainer = document.getElementById('plots-container');
            const currency = document.getElementById('currency-select').value;
            const timeframe = document.getElementById('timeframe-select').value;
            
            btn.disabled = true;
            statusEl.style.display = 'block';
            statusEl.className = 'status loading';
            statusEl.innerHTML = '<span class="spinner"></span>Starting parallel analysis for ' + currency + '/' + timeframe + '...';
            plotsContainer.innerHTML = '<p>⚙️ Processing...</p>';
            
            try {
                // Start analysis job
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ currency, timeframe })
                });
                
                const jobInfo = await response.json();
                if (!jobInfo.success) throw new Error(jobInfo.error || 'Failed to start');
                
                const jobId = jobInfo.job_id;
                statusEl.innerHTML = '<span class="spinner"></span>Running parallel plot generation (Job: ' + jobId.substring(0, 8) + '...)';
                
                // Poll for completion
                const poll = async () => {
                    const statusResponse = await fetch('/api/job/' + jobId);
                    const status = await statusResponse.json();
                    
                    if (status.status === 'running') {
                        statusEl.innerHTML = '<span class="spinner"></span>' + (status.progress || 'Processing...'); 
                        setTimeout(poll, 1000);
                    } else if (status.status === 'completed') {
                        const result = status.result;
                        statusEl.className = 'status success';
                        statusEl.innerHTML = '✓ Analysis complete! Generated ' + result.plots.length + ' plots in ' + result.execution_time + 's (parallel speedup!)';
                        
                        plotsContainer.innerHTML = result.plots.map(p => 
                            '<img src="' + p + '?t=' + Date.now() + '" class="plot-img">'
                        ).join('');
                        btn.disabled = false;
                    } else {
                        throw new Error(status.error || 'Analysis failed');
                    }
                };
                setTimeout(poll, 1000);
                
            } catch (error) {
                statusEl.className = 'status error';
                statusEl.innerHTML = '✗ Error: ' + error.message;
                plotsContainer.innerHTML = '<p>❌ ' + error.message + '</p>';
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>"""
        
        html_bytes = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(html_bytes)))
        self.end_headers()
        self.wfile.write(html_bytes)
    
    def serve_currencies(self):
        """Serve available currencies"""
        currencies = ['BTC', 'ETH', 'SOL']
        self.send_json_response({'currencies': currencies})
    
    def serve_job_status(self, job_id):
        """Serve job status"""
        with job_lock:
            if job_id not in jobs:
                self.send_json_response({'success': False, 'error': 'Job not found'}, status=404)
                return
            
            job = jobs[job_id]
            if job['status'] == 'running':
                self.send_json_response({'status': 'running', 'progress': job.get('progress', 'Processing...')})
            elif job['status'] == 'completed':
                self.send_json_response({'status': 'completed', 'result': job['result']})
            else:
                self.send_json_response({'status': 'failed', 'error': job.get('error', 'Unknown error')})
    
    def handle_analyze(self):
        """Handle analysis request - start async parallel job"""
        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())
            
            currency = data.get('currency', 'BTC').upper()
            timeframe = data.get('timeframe', '5m').lower()
            
            if currency not in ['BTC', 'ETH', 'SOL']:
                self.send_json_response({'success': False, 'error': f'Invalid currency: {currency}'}, status=400)
                return
            
            job_id = str(uuid.uuid4())
            
            with job_lock:
                jobs[job_id] = {'status': 'running', 'currency': currency, 'timeframe': timeframe, 'start_time': time.time(), 'progress': 'Starting...'}
            
            print(f"[JOB] Created parallel job {job_id[:8]} for {currency}/{timeframe}")
            
            thread = threading.Thread(target=self.run_parallel_analysis, args=(job_id, currency, timeframe), daemon=True)
            thread.start()
            
            self.send_json_response({'success': True, 'job_id': job_id, 'message': 'Parallel analysis started'})
            
        except Exception as e:
            self.send_json_response({'success': False, 'error': str(e)}, status=500)
    
    def run_parallel_analysis(self, job_id, currency, timeframe):
        """Run analysis with parallel plot generation"""
        try:
            start_time = time.time()
            output_dir = f'{OUTPUT_DIR}/{currency.lower()}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Prepare data (sequential - GPU computation)
            with job_lock:
                jobs[job_id]['progress'] = 'Preparing data & running GPU decomposition...'
            
            prep_result = subprocess.run(
                [PYTHON_BIN, 'prepare_data.py', currency, timeframe],
                capture_output=True, text=True, timeout=60
            )
            
            if prep_result.returncode != 0:
                with job_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = f'Data preparation failed: {prep_result.stderr}'
                return
            
            # Step 2: Generate all plots using original script (includes all 6 plots)
            with job_lock:
                jobs[job_id]['progress'] = 'Generating all plots...'
            
            plot_result = subprocess.run(
                [PYTHON_BIN, 'gpu_wavelet_gpu_plot.py', currency, timeframe],
                capture_output=True, text=True, timeout=120
            )
            
            if plot_result.returncode != 0:
                with job_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = f'Plot generation failed: {plot_result.stderr}'
                return
            
            total_time = time.time() - start_time
            
            # Collect plot files (original script generates all 6)
            plot_files = [
                '01_main_overview.png',
                '02a_progressive_approximations.png',
                '02b_frequency_bands.png',
                '03_anomaly_detection.png',
                '04_trading_signals.png',
                '05_statistics_dashboard.png'
            ]
            plots = [f'/plots/{currency.lower()}/{f}' for f in plot_files if os.path.exists(f'{output_dir}/{f}')]
            
            # Update job with results
            with job_lock:
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['result'] = {
                    'success': True,
                    'currency': currency,
                    'plots': plots,
                    'execution_time': f'{total_time:.2f}',
                    'message': 'Parallel generation complete'
                }
            
            print(f"[JOB] {job_id[:8]} completed in {total_time:.2f}s")
            
        except Exception as e:
            with job_lock:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = str(e)
            print(f"[JOB] {job_id[:8]} failed: {e}")
    
    def serve_image(self, path):
        """Serve plot images"""
        relative_path = path[7:]
        file_path = os.path.join(OUTPUT_DIR, relative_path)
        
        if not os.path.exists(file_path):
            self.send_error(404)
            return
        
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.send_header('Cache-Control', 'max-age=3600')
        self.end_headers()
        
        with open(file_path, 'rb') as f:
            self.wfile.write(f.read())
    
    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        """Override to add timestamp"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {format % args}")


def main():
    """Start the parallel web server"""
    server_address = ('', PORT)
    httpd = ThreadingHTTPServer(server_address, ParallelWaveletHandler)
    
    print("=" * 70)
    print("PARALLEL GPU WAVELET ANALYSIS WEB SERVER")
    print("=" * 70)
    print(f"\n✓ Server started on http://localhost:{PORT}")
    print(f"✓ Plot generation: PARALLEL (6 separate processes)")
    print(f"✓ Output directory: {OUTPUT_DIR}/")
    print(f"\nOpen your browser: http://localhost:{PORT}")
    print("\nPress Ctrl+C to stop\n")
    print("=" * 70)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped")
        httpd.shutdown()


if __name__ == '__main__':
    main()
