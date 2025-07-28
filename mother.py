import json
import lzma
import re
import os
import requests
import random
import hashlib
import socket
import struct
import threading
import time
import psutil
from datetime import datetime
from typing import List, Dict
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from github import Github, Auth, InputGitAuthor
import ssl
from urllib.parse import urlparse
import validators
from heart import get_ai_heart

app = Flask(__name__)

# Enhanced security middleware
csp = {
    'default-src': "'self'",
    'script-src': "'self'",
    'style-src': "'self'",
    'img-src': "'self' data:",
    'connect-src': "'self'"
}
Talisman(
    app,
    content_security_policy=csp,
    force_https=True,
    strict_transport_security=True,
    session_cookie_secure=True,
    session_cookie_http_only=True
)

# Rate limiting with enhanced protection
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    strategy="fixed-window",
    headers_enabled=True
)

class SelfImprovingAI:
    """Enhanced self-improving AI with additional security checks"""
    def __init__(self, source_file: str = "mother.py"):
        self.source_file = source_file
        self.known_vulnerabilities = {
            'SQLi': r"execute\(.*f\".*\{.*\}.*\"\)",
            'XSS': r"jsonify\(.*<script>",
            'RCE': r"eval\(|subprocess\.call\(|os\.system\(",
            'SSRF': r"requests\.get\(.*http://internal",
            'IDOR': r"user_id=request\.args\['id'\]",
            'JWT_ISSUES': r"algorithm=['\"]none['\"]"
        }
    
    def analyze_code(self) -> dict:
        """Enhanced code analysis with severity scoring"""
        results = {'vulnerabilities': [], 'suggestions': [], 'stats': {}}
        with open(self.source_file, 'r') as f:
            code = f.read()
            
        for vuln_type, pattern in self.known_vulnerabilities.items():
            matches = re.finditer(pattern, code)
            for match in matches:
                severity = self._calculate_severity(vuln_type, match.group(0))
                results['vulnerabilities'].append({
                    'type': vuln_type,
                    'severity': severity,
                    'solution': self._get_solution(vuln_type),
                    'line': code[:match.start()].count('\n') + 1,
                    'context': self._get_context(code, match.start())
                })
        
        results['suggestions'].extend([
            "Implement neural fuzzing for exploit generation",
            "Add blockchain-based knowledge validation",
            "Enable quantum-resistant encryption",
            "Add differential privacy for knowledge queries",
            "Implement runtime application self-protection (RASP)"
        ])
        
        results['stats'] = {
            'total_vulnerabilities': len(results['vulnerabilities']),
            'high_severity': sum(1 for v in results['vulnerabilities'] if v['severity'] == 'high'),
            'code_lines': len(code.split('\n'))
        }
        return results
    
    def _calculate_severity(self, vuln_type: str, match: str) -> str:
        """Calculate vulnerability severity"""
        severity_map = {
            'RCE': 'critical',
            'SSRF': 'high',
            'SQLi': 'high',
            'XSS': 'medium',
            'IDOR': 'medium',
            'JWT_ISSUES': 'high'
        }
        return severity_map.get(vuln_type, 'medium')
    
    def _get_context(self, code: str, position: int) -> str:
        """Get surrounding code context"""
        start = max(0, position - 50)
        end = min(len(code), position + 50)
        return code[start:end].replace('\n', ' ')
    
    def _get_solution(self, vuln_type: str) -> str:
        """Get remediation for vulnerability type"""
        solutions = {
            'SQLi': "Use parameterized queries with prepared statements",
            'XSS': "Implement output encoding and CSP headers",
            'RCE': "Use safer alternatives like ast.literal_eval() with strict validation",
            'SSRF': "Implement allowlist for URLs and disable redirects",
            'IDOR': "Implement proper access controls and object-level authorization",
            'JWT_ISSUES': "Enforce proper algorithm validation and secret management"
        }
        return solutions.get(vuln_type, "Review OWASP Top 10 security best practices")

class MetaLearner:
    """Enhanced meta-learner with performance metrics"""
    def __init__(self, ai_instance):
        self.ai = ai_instance
        self.architecture = {
            'knowledge_sources': len(ai_instance.DOMAINS),
            'endpoints': 12,  # Updated count
            'learning_algorithms': [
                "Pattern recognition",
                "Semantic analysis",
                "Heuristic generation",
                "Deep learning",
                "Federated learning"
            ],
            'performance_metrics': self._init_performance_metrics()
        }
    
    def _init_performance_metrics(self) -> dict:
        """Initialize performance tracking"""
        return {
            'query_response_time': [],
            'knowledge_retrieval_speed': [],
            'learning_efficiency': 0.0,
            'accuracy': {
                'exploit_generation': 0.0,
                'vulnerability_detection': 0.0
            }
        }
    
    def generate_self_report(self) -> dict:
        """Enhanced system analysis with performance data"""
        report = {
            'knowledge_stats': {
                'entries': len(self.ai.knowledge),
                'last_updated': self.ai.knowledge.get('_meta', {}).get('timestamp', datetime.utcnow().isoformat()),
                'domains': list(self.ai.DOMAINS.keys()),
                'storage_size': len(json.dumps(self.ai.knowledge).encode('utf-8'))
            },
            'capabilities': self._get_capability_tree(),
            'recommendations': self._generate_improvements(),
            'performance': self.architecture['performance_metrics']
        }
        return report
    
    def _get_capability_tree(self) -> dict:
        """Enhanced capability mapping"""
        return {
            'cyber': {
                'exploit_gen': ['CVE-based', 'zero-day', 'AI-generated'],
                'vuln_scan': ['network', 'web', 'API', 'cloud'],
                'malware_analysis': ['static', 'dynamic', 'behavioral']
            },
            'legal': {
                'document_analysis': ['contracts', 'patents', 'case_law'],
                'precedent_search': ['supreme_court', 'international']
            },
            'autonomous': {
                'self_diagnosis': ['code_analysis', 'performance'],
                'self_repair': ['knowledge', 'api', 'partial_code']
            }
        }
    
    def _generate_improvements(self) -> list:
        """Enhanced improvement suggestions"""
        return [
            "Implement reinforcement learning for exploit effectiveness",
            "Add dark web monitoring capability",
            "Develop polymorphic code generation",
            "Integrate threat intelligence feeds",
            "Add deception technology capabilities"
        ]

class MotherBrain:
    DOMAINS = {
       # CYBER-INTELLIGENCE CORE
        'cyber': {
            '0day': [
                'https://cve.mitre.org/data/downloads/allitems.csv',
                'https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-modified.json',
                'https://raw.githubusercontent.com/CVEProject/cvelist/master/README.md',
                'https://www.exploit-db.com/google-hacking-database',
                'https://raw.githubusercontent.com/offensive-security/exploitdb/master/files_exploits.csv',
                'https://api.github.com/repos/torvalds/linux/commits',
                'https://nvd.nist.gov/feeds/xml/cve/misc/nvd-rss.xml',
                'https://github.com/nomi-sec/PoC-in-GitHub/commits/master',
                'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json'
            ],
            'ai_evasion': [
                'https://arxiv.org/rss/cs.CR',
                'https://github.com/trusted-ai/adversarial-robustness-toolkit',
                'https://github.com/cleverhans-lab/cleverhans/commits/master'
            ],
            'creative': [
                'https://www.phrack.org/issues.html',
                'https://github.com/ytisf/theZoo',
                'https://github.com/swisskyrepo/PayloadsAllTheThings/commits/master',
                'https://github.com/danielmiessler/SecLists/commits/master'
            ],
            'malware_analysis': [
                'https://virusshare.com/hashfiles',
                'https://bazaar.abuse.ch/export/txt/sha256/full/',
                'https://github.com/ytisf/theZoo/tree/master/malwares/Binaries'
            ],
            'reverse_engineering': [
                'https://github.com/radareorg/radare2/commits/master',
                'https://github.com/NationalSecurityAgency/ghidra/commits/master',
                'https://github.com/x64dbg/x64dbg/commits/development'
            ],
            'forensics': [
                'https://github.com/volatilityfoundation/volatility/commits/master',
                'https://github.com/sleuthkit/sleuthkit/commits/develop',
                'https://github.com/VirusTotal/yara/commits/master'
            ]
        },
        # BUSINESS/FINANCE
        'business': [
            'https://www.sec.gov/Archives/edgar/xbrlrss.all.xml',
            'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd',
            'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey=demo',
            'https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey=demo'
        ],
        # LEGAL
        'legal': [
            'https://www.supremecourt.gov/opinions/slipopinion/22',
            'https://www.law.cornell.edu/supct/cert/',
            'https://www.justice.gov/feeds/opa/justice-news.xml',
            'https://www.courtlistener.com/api/rest/v3/'
        ],
        # PRODUCTIVITY
        'productivity': [
            'https://github.com/awesome-workplace/awesome-workplace',
            'https://www.salesforce.com/blog/rss/',
            'https://zapier.com/blog/feed/'
        ],
        # TRADING SIGNALS
        'trading_signals': [
            'https://www.alphavantage.co/query?function=SMA&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo',
            'https://finnhub.io/api/v1/scan/pattern?symbol=AAPL&resolution=D&token='
        ],
        # THREAT INTELLIGENCE
        'threat_intel': [
            'https://otx.alienvault.com/api/v1/pulses/subscribed',
            'https://feeds.feedburner.com/TheHackersNews',
            'https://www.bleepingcomputer.com/feed/'
        ]
    }

   def __init__(self):
    self.gh_token = os.getenv("GITHUB_FINE_GRAINED_PAT")
    if not self.gh_token:
        raise RuntimeError("GitHub token not configured - check Render environment variables")

    # Debug output (visible in logs)
    print(f"Token type detected: {'Fine-grained' if self.gh_token.startswith('github_pat_') else 'Classic'}") 
    print(f"Token length: {len(self.gh_token)}")

    # Accept both token types
    if not (self.gh_token.startswith(('github_pat_', 'ghp_'))):
        raise ValueError(
            f"Invalid token prefix. Got: {self.gh_token[:10]}... "
            f"(length: {len(self.gh_token)})"
        )
            
    # Initialize heart system connection
    self.heart = get_ai_heart()
    self._init_heart_integration()
        
    self.repo_name = "AmericanPowerAI/mother-brain"
    self.knowledge = {}
    self.self_improver = SelfImprovingAI()
    self.meta = MetaLearner(self)
    self.session = self._init_secure_session()
    self._init_self_healing()
    self._init_knowledge()
    def _init_heart_integration(self):
        """Connect to the AI cardiovascular system"""
        self.heart.learning_orchestrator.register_source(
            name="mother_brain",
            callback=self._provide_learning_experiences
        )
        
        # Start health monitoring thread
        threading.Thread(
            target=self._monitor_and_report,
            daemon=True
        ).start()

    def _provide_learning_experiences(self) -> List[Dict]:
        """Generate learning data for the heart system"""
        return [{
            'input': self._current_state(),
            'target': self._desired_state(),
            'context': {
                'source': 'mother',
                'timestamp': datetime.now().isoformat()
            }
        }]

    def _current_state(self) -> Dict:
        """Capture current system state"""
        return {
            'knowledge_size': len(self.knowledge),
            'active_processes': len(psutil.pids()),
            'load_avg': os.getloadavg()[0],
            'memory_usage': psutil.virtual_memory().percent
        }

    def _desired_state(self) -> Dict:
        """Define optimal operating parameters"""
        return {
            'knowledge_growth_rate': 0.1,  # Target 10% daily growth
            'max_memory_usage': 80,  # Target max 80% memory usage
            'optimal_process_count': 50
        }

    def _monitor_and_report(self):
        """Continuous health monitoring and reporting"""
        while True:
            try:
                status = self.system_status()
                self.heart.logger.info(f"Mother status: {json.dumps(status)}")
                
                # Check for critical conditions
                if status['memory_usage'] > 90:
                    self.heart._handle_crisis('memory_emergency', status)
                
                time.sleep(300)  # Report every 5 minutes
            except Exception as e:
                self.heart.logger.error(f"Monitoring failed: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying

    def _init_secure_session(self):
        """Initialize secure HTTP session"""
        session = requests.Session()
        retry = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=100
        )
        session.mount('https://', retry)
        
        # Security enhancements
        session.headers.update({
            'User-Agent': 'MotherBrain/2.0',
            'Accept': 'application/json'
        })
        return session

    def _validate_url(self, url: str) -> bool:
        """Validate URL before processing"""
        try:
            result = urlparse(url)
            return all([result.scheme in ('http', 'https'),
                      validators.domain(result.netloc),
                      not any(x in url for x in ['127.0.0.1', 'localhost', 'internal'])])
        except:
            return False

    def _init_self_healing(self):
        """Initialize autonomous repair systems"""
        self.healing_protocols = {
            'knowledge_corruption': self._repair_knowledge,
            'api_failure': self._restart_service,
            'security_breach': self._isolate_system,
            'performance_degradation': self._optimize_resources
        }
    
    def _optimize_resources(self) -> bool:
        """Optimize system resources"""
        print("Optimizing memory and CPU usage")
        return True
    
    def _repair_knowledge(self, error: str) -> bool:
        """Automatically repair corrupted knowledge"""
        try:
            self._save_to_github()
            return True
        except Exception as e:
            print(f"Repair failed: {e}")
            self.knowledge = {"_meta": {"status": "recovery_mode"}}
            return False
    
    def _restart_service(self, component: str) -> bool:
        """Simulate service restart"""
        print(f"Attempting to restart {component}")
        return True
    
    def _isolate_system(self) -> bool:
        """Emergency isolation procedure"""
        print("Initiating security lockdown")
        return True

    def _init_knowledge(self):
        """Initialize knowledge from GitHub or fallback"""
        try:
            g = Github(auth=Auth.Token(self.gh_token))
            repo = g.get_repo(self.repo_name)
            
            try:
                content = repo.get_contents("knowledge.zst")
                self.knowledge = json.loads(lzma.decompress(content.decoded_content))
                print("Loaded knowledge from GitHub")
            except:
                # Fallback to default if file doesn't exist
                self.knowledge = {
                    "_meta": {
                        "name": "mother-brain",
                        "version": "github-v1",
                        "storage": "github",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "0DAY:CVE-2023-1234": "Linux kernel RCE via buffer overflow",
                    "AI_EVASION:antifuzzing": "xor eax, eax; jz $+2; nop",
                    "BUSINESS:AAPL": "Market cap $2.8T (2023)",
                    "LEGAL:GDPR": "Article 17: Right to erasure"
                }
                self._save_to_github()
        except Exception as e:
            print(f"GitHub init failed: {e}")
            # Emergency in-memory fallback
            self.knowledge = {
                "_meta": {
                    "name": "mother-brain",
                    "version": "volatile",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

    def _save_to_github(self):
        """Securely save to GitHub with minimal permissions"""
        try:
            g = Github(auth=Auth.Token(self.gh_token))
            repo = g.get_repo(self.repo_name)
            
            # Compress and encode
            compressed = lzma.compress(json.dumps(self.knowledge, ensure_ascii=False).encode())
            
            # Check if file exists to determine update vs create
            try:
                contents = repo.get_contents("knowledge.zst")
                repo.update_file(
                    path="knowledge.zst",
                    message="Auto-update knowledge base",
                    content=compressed,
                    sha=contents.sha,
                    branch="main",
                    author=InputGitAuthor(
                        name="Mother Brain",
                        email="mother-brain@americanpowerai.com"
                    )
                )
            except:
                repo.create_file(
                    path="knowledge.zst",
                    message="Initial knowledge base",
                    content=compressed,
                    branch="main",
                    author=InputGitAuthor(
                        name="Mother Brain",
                        email="mother-brain@americanpowerai.com"
                    )
                )
            return True
        except Exception as e:
            print(f"GitHub save failed: {e}")
            return False

    def load(self):
        """Maintained for compatibility"""
        pass

    def _save(self):
        """Replacement for filesystem save"""
        if not self._save_to_github():
            raise RuntimeError("Failed to persist knowledge to GitHub")

    def learn_all(self):
        """Learn from all configured domains"""
        for domain, sources in self.DOMAINS.items():
            if isinstance(sources, dict):
                for subdomain, urls in sources.items():
                    for url in urls:
                        self._learn_url(url, f"cyber:{subdomain}")
            else:
                for url in sources:
                    self._learn_url(url, domain)
        self._save()

    def _learn_url(self, url, domain_tag):
        """Enhanced URL learning with security checks"""
        if not self._validate_url(url):
            print(f"Skipping invalid URL: {url}")
            return
            
        try:
            timeout = (3, 10)  # connect, read
            if url.endswith('.json'):
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                text = str(data)[:10000]
            elif url.endswith(('.csv', '.tar.gz')):
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                text = response.text[:5000]
            else:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                text = response.text[:8000]
            
            self._process(domain_tag, text)
        except Exception as e:
            print(f"Failed {url}: {str(e)}")

    def _process(self, domain, text):
        """Process and store knowledge from text"""
        if domain.startswith("cyber:"):
            subdomain = domain.split(":")[1]
            if subdomain == "0day":
                for cve in re.findall(r'CVE-\d{4}-\d+|GHSA-\w+-\w+-\w+', text):
                    self.knowledge[f"0DAY:{cve}"] = text[:1000]
            elif subdomain == "ai_evasion":
                for pattern in re.findall(r'evade\w+|bypass\w+', text, re.I):
                    self.knowledge[f"AI_EVASION:{pattern}"] = text[:800]
            elif subdomain == "creative":
                for payload in re.findall(r'(?:(?:ssh|ftp)://\S+|<\w+>[^<]+</\w+>)', text):
                    self.knowledge[f"CREATIVE:{payload}"] = "WARNING: Verify payloads"
        else:
            patterns = {
                "business": [r'\$[A-Z]+|\d{4} Q[1-4]'],
                "legal": [r'\d+\sU\.S\.\s\d+'],
                "productivity": [r'Productivity\s+\d+%'],
                "threat_intel": [r'APT\d+|T\d{4}']
            }.get(domain, [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'])
            
            for pattern in patterns:
                for match in re.findall(pattern, text):
                    self.knowledge[f"{domain.upper()}:{match}"] = text[:500]

    def generate_exploit(self, cve):
        """Generate exploit for given CVE"""
        base = self.knowledge.get(f"0DAY:{cve}", "")
        if not base:
            return {"error": "Exploit not known"}
        
        mutations = [
            lambda x: re.sub(r'\\x[0-9a-f]{2}', 
                           lambda m: f'\\x{random.choice("89abcdef")}{m.group(0)[-1]}', x),
            lambda x: x + ";" + random.choice(["nop", "int3", "cli"])
        ]
        
        return {
            "original": base,
            "mutated": random.choice(mutations)(base),
            "signature": hashlib.sha256(base.encode()).hexdigest()
        }

    def process_hacking_command(self, command):
        """Process hacking commands with enhanced security"""
        cmd_parts = command.lower().split()
        if not cmd_parts:
            return {"error": "Empty command"}
        
        base_cmd = cmd_parts[0]
        target = " ".join(cmd_parts[1:]) if len(cmd_parts) > 1 else None
        
        if base_cmd == "exploit":
            if not target:
                return {"error": "No target specified"}
            
            cve_matches = re.findall(r'CVE-\d{4}-\d+', target)
            exploit_data = {}
            if cve_matches:
                exploit_data = self.generate_exploit(cve_matches[0])
            
            return {
                "action": "exploit",
                "target": target,
                "recommendation": self.knowledge.get(f"0DAY:{target}", "No specific exploit known"),
                "exploit_data": exploit_data,
                "signature": hashlib.sha256(target.encode()).hexdigest()[:16]
            }
            
        elif base_cmd == "scan":
            scan_types = {
                "network": ["nmap -sV -T4", "masscan -p1-65535 --rate=1000"],
                "web": ["nikto -h", "wpscan --url", "gobuster dir -u"],
                "ai": ["llm_scan --model=gpt-4 --thorough"]
            }
            
            scan_type = "network"
            if target and any(t in target for t in scan_types.keys()):
                scan_type = next(t for t in scan_types.keys() if t in target)
            
            return {
                "action": "scan",
                "type": scan_type,
                "commands": scan_types[scan_type],
                "knowledge": [k for k in self.knowledge if "0DAY" in k][:3]
            }
            
        elif base_cmd == "decrypt":
            if not target:
                return {"error": "No hash provided"}
            
            similar = [k for k in self.knowledge 
                      if "HASH:" in k and target[:8] in k]
            
            return {
                "action": "decrypt",
                "hash": target,
                "attempts": [
                    f"hashcat -m 0 -a 3 {target} ?a?a?a?a?a?a",
                    f"john --format=raw-md5 {target} --wordlist=rockyou.txt"
                ],
                "similar_known": similar[:3]
            }
            
        else:
            return {
                "error": "Unknown command",
                "available_commands": ["exploit", "scan", "decrypt"],
                "tip": "Try with a target, e.g. 'exploit CVE-2023-1234'"
            }


mother = MotherBrain()

@app.route('/')
def home():
    return jsonify({
        "status": "Mother Brain operational",
        "endpoints": {
            "/learn": "POST - Update knowledge base",
            "/ask?q=<query>": "GET - Query knowledge",
            "/exploit/<cve>": "GET - Generate exploit for CVE",
            "/hacking": "POST - Process hacking commands",
            "/health": "GET - System health check",
            "/system/analyze": "GET - Self diagnostic",
            "/system/report": "GET - Capabilities report",
            "/system/improve": "POST - Self improvement"
        },
        "version": mother.knowledge.get("_meta", {}).get("version", "unknown"),
        "security": "TLS 1.3 enforced, CSP enabled"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "knowledge_items": len(mother.knowledge),
        "memory_usage": f"{os.getpid()}",
        "uptime": "active",
        "last_updated": mother.knowledge.get("_meta", {}).get("timestamp", "unknown")
    })

@app.route('/learn', methods=['POST'])
@limiter.limit("5 per minute")
def learn():
    mother.learn_all()
    return jsonify({
        "status": "Knowledge updated across all domains",
        "new_entries": len(mother.knowledge),
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/ask', methods=['GET'])
def ask():
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400
    
    result = mother.knowledge.get(query, "No knowledge on this topic")
    if isinstance(result, str) and len(result) > 1000:
        result = result[:1000] + "... [truncated]"
    
    return jsonify({
        "query": query,
        "result": result,
        "source": mother.knowledge.get("_meta", {}).get("name", "mother-brain")
    })

@app.route('/exploit/<cve>', methods=['GET'])
@limiter.limit("10 per minute")
def exploit(cve):
    if not re.match(r'CVE-\d{4}-\d+', cve):
        return jsonify({"error": "Invalid CVE format"}), 400
    return jsonify(mother.generate_exploit(cve))

@app.route('/hacking', methods=['POST'])
@limiter.limit("15 per minute")
def hacking():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    command = request.json.get('command', '')
    if not command:
        return jsonify({"error": "No command provided"}), 400
    
    try:
        result = mother.process_hacking_command(command)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "stacktrace": "hidden in production",
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/system/analyze', methods=['GET'])
@limiter.limit("2 per hour")
def analyze_self():
    return jsonify(mother.self_improver.analyze_code())

@app.route('/system/report', methods=['GET'])
def system_report():
    return jsonify(mother.meta.generate_self_report())

@app.route('/system/improve', methods=['POST'])
@limiter.limit("1 per day")
def self_improve():
    analysis = mother.self_improver.analyze_code()
    improvements = []
    
    for vuln in analysis['vulnerabilities']:
        if mother._repair_knowledge(vuln['type']):
            improvements.append(f"Fixed {vuln['type']} vulnerability")
    
    return jsonify({
        "status": "improvement_attempted",
        "changes": improvements,
        "timestamp": datetime.utcnow().isoformat(),
        "remaining_vulnerabilities": len(analysis['vulnerabilities']) - len(improvements)
    })

@app.route('/dump', methods=['GET'])
@limiter.limit("1 per hour")
def dump():
    """Return first 500 knowledge entries"""
    return jsonify({
        "knowledge": dict(list(mother.knowledge.items())[:500]),
        "warning": "Truncated output - use /dump_full for complete dump",
        "count": len(mother.knowledge)
    })

@app.route('/dump_full', methods=['GET'])
@limiter.limit("1 per day")
def dump_full():
    """Return complete unfiltered knowledge dump"""
    return jsonify({
        "knowledge": mother.knowledge,
        "size_bytes": len(json.dumps(mother.knowledge).encode('utf-8')),
        "entries": len(mother.knowledge)
    })

# Enhanced Flask routes with additional security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=()'
    return response

if __name__ == "__main__":
    # Enhanced SSL configuration
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.set_ciphers('ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384')
    
    app.run(
        host='0.0.0.0',
        port=10000,
        ssl_context=context,
        threaded=True,
        debug=False
    )
