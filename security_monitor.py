import psutil
import json
import hashlib
import socket
import threading
from datetime import datetime
from flask import Flask, jsonify
import requests
import logging
from typing import Dict, List, Optional
import re
import os
import sys

class SecurityMonitor:
    def __init__(self, mother_url: str, api_key: str):
        self.mother_url = mother_url
        self.api_key = api_key
        self.baseline = self._establish_baseline()
        self.anomaly_threshold = 0.85  # Statistical threshold for anomalies
        self.logger = self._setup_logging()
        self.running = True
        self.whitelist = self._load_whitelist()
        
        # Initialize monitoring threads
        self.monitor_threads = [
            threading.Thread(target=self._monitor_memory),
            threading.Thread(target=self._monitor_network),
            threading.Thread(target=self._monitor_processes),
            threading.Thread(target=self._monitor_filesystem)
        ]

    def _setup_logging(self) -> logging.Logger:
        """Configure secure logging with rotation"""
        logger = logging.getLogger('SecurityMonitor')
        logger.setLevel(logging.INFO)
        
        handler = logging.handlers.RotatingFileHandler(
            'security.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def _establish_baseline(self) -> Dict:
        """Create baseline system profile"""
        return {
            'memory': psutil.virtual_memory().percent,
            'cpu': psutil.cpu_percent(),
            'network': self._get_network_stats(),
            'processes': len(psutil.pids()),
            'files': self._scan_critical_files()
        }

    def _load_whitelist(self) -> Dict:
        """Load approved process/file hashes"""
        try:
            response = requests.get(
                f"{self.mother_url}/api/whitelist",
                headers={'Authorization': f'Bearer {self.api_key}'},
                timeout=5
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Whitelist load failed: {str(e)}")
            return {
                'processes': ['python', 'security_monitor'],
                'files': self._get_core_hashes()
            }

    def _get_core_hashes(self) -> Dict:
        """Generate SHA-256 hashes of critical files"""
        core_files = ['mother.py', 'child.py', 'security_monitor.py']
        return {f: self._file_hash(f) for f in core_files if os.path.exists(f)}

    def _file_hash(self, path: str) -> str:
        """Calculate file hash with salt"""
        salt = os.urandom(16)
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(salt + chunk)
        return hasher.hexdigest()

    def start_monitoring(self):
        """Start all monitoring threads"""
        for thread in self.monitor_threads:
            thread.daemon = True
            thread.start()
        self.logger.info("Security monitoring started")

    def stop_monitoring(self):
        """Graceful shutdown"""
        self.running = False
        for thread in self.monitor_threads:
            thread.join(timeout=1)
        self.logger.info("Security monitoring stopped")

    def _monitor_memory(self):
        """Detect memory anomalies/injections"""
        while self.running:
            current = psutil.virtual_memory().percent
            if abs(current - self.baseline['memory']) > 15:  # 15% deviation
                alert = {
                    'type': 'memory_anomaly',
                    'current': current,
                    'baseline': self.baseline['memory'],
                    'timestamp': datetime.utcnow().isoformat()
                }
                self._send_alert(alert)
            threading.Event().wait(5)

    def _monitor_network(self):
        """Detect suspicious network activity"""
        while self.running:
            current = self._get_network_stats()
            if current['connections'] > self.baseline['network']['connections'] * 2:
                alert = {
                    'type': 'network_flood',
                    'current': current,
                    'baseline': self.baseline['network'],
                    'timestamp': datetime.utcnow().isoformat()
                }
                self._send_alert(alert)
            threading.Event().wait(10)

    def _monitor_processes(self):
        """Detect unauthorized processes"""
        while self.running:
            current_procs = {p.name() for p in psutil.process_iter(['name'])}
            unknown = current_procs - set(self.whitelist.get('processes', []))
            
            if unknown:
                alert = {
                    'type': 'unauthorized_process',
                    'processes': list(unknown),
                    'timestamp': datetime.utcnow().isoformat()
                }
                self._send_alert(alert)
            threading.Event().wait(30)

    def _monitor_filesystem(self):
        """Detect file tampering"""
        while self.running:
            current_hashes = self._get_core_hashes()
            for file, known_hash in self.whitelist.get('files', {}).items():
                if current_hashes.get(file) != known_hash:
                    alert = {
                        'type': 'file_tampering',
                        'file': file,
                        'current_hash': current_hashes.get(file),
                        'known_hash': known_hash,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    self._send_alert(alert)
            threading.Event().wait(60)

    def _get_network_stats(self) -> Dict:
        """Get current network metrics"""
        conns = psutil.net_connections()
        return {
            'connections': len(conns),
            'ports': {c.laddr.port for c in conns if c.status == 'LISTEN'},
            'bandwidth': psutil.net_io_counters().bytes_sent + 
                        psutil.net_io_counters().bytes_recv
        }

    def _scan_critical_files(self) -> Dict:
        """Baseline scan of system files"""
        return {
            '/etc/passwd': self._file_hash('/etc/passwd'),
            '/etc/shadow': self._file_hash('/etc/shadow') if os.access('/etc/shadow', os.R_OK) else None
        }

    def _send_alert(self, alert_data: Dict):
        """Securely transmit alerts to Mother Brain"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'X-Security-Hash': hashlib.sha256(
                    json.dumps(alert_data).encode()
                ).hexdigest()
            }
            
            requests.post(
                f"{self.mother_url}/api/alerts",
                json=alert_data,
                headers=headers,
                timeout=3
            )
            self.logger.warning(f"Alert sent: {alert_data['type']}")
        except Exception as e:
            self.logger.error(f"Alert failed: {str(e)}")

    def detect_code_injection(self, code: str) -> bool:
        """Static analysis for malicious patterns"""
        patterns = [
            r'(exec|eval|subprocess)\(.*\)',
            r'__import__\(.*\)',
            r'os\.system\(.*\)',
            r'pickle\.loads\(.*\)'
        ]
        return any(re.search(p, code) for p in patterns)

# Flask API for local queries
app = Flask(__name__)
monitor = None

@app.route('/security/status', methods=['GET'])
def get_status():
    return jsonify({
        'memory': psutil.virtual_memory().percent,
        'cpu': psutil.cpu_percent(),
        'network': monitor._get_network_stats(),
        'alerts': monitor.logger.handlers[0].baseFilename if monitor else None
    })

@app.route('/security/scan', methods=['POST'])
def scan_file():
    if not request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    temp_path = f"/tmp/{file.filename}"
    file.save(temp_path)
    
    result = {
        'hash': monitor._file_hash(temp_path),
        'injection_detected': monitor.detect_code_injection(
            file.stream.read().decode('utf-8', errors='ignore')
        ),
        'path': temp_path
    }
    
    os.unlink(temp_path)
    return jsonify(result)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python security_monitor.py <mother_url> <api_key>")
        sys.exit(1)
        
    monitor = SecurityMonitor(sys.argv[1], sys.argv[2])
    monitor.start_monitoring()
    
    # Start Flask in separate thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host='127.0.0.1', port=5001)
    )
    flask_thread.daemon = True
    flask_thread.start()
    
    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        sys.exit(0)
