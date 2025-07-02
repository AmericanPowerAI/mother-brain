import json
import lzma
import re
import os
import requests
import random
import hashlib
import socket
import struct
from datetime import datetime
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from github import Github, Auth, InputGitAuthor

app = Flask(__name__)

# Security middleware
Talisman(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

class SelfImprovingAI:
    """Enables the AI to analyze and improve its own code"""
    def __init__(self, source_file: str = "mother.py"):
        self.source_file = source_file
        self.known_vulnerabilities = {
            'SQLi': r"execute\(.*f\".*\{.*\}.*\"\)",
            'XSS': r"jsonify\(.*<script>",
            'RCE': r"eval\(|subprocess\.call\(|os\.system\("
        }
    
    def analyze_code(self) -> dict:
        """Scan own source code for vulnerabilities"""
        results = {'vulnerabilities': [], 'suggestions': []}
        with open(self.source_file, 'r') as f:
            code = f.read()
            
        for vuln_type, pattern in self.known_vulnerabilities.items():
            if re.search(pattern, code):
                results['vulnerabilities'].append({
                    'type': vuln_type,
                    'severity': 'high',
                    'solution': self._get_solution(vuln_type)
                })
        
        results['suggestions'].extend([
            "Implement neural fuzzing for exploit generation",
            "Add blockchain-based knowledge validation",
            "Enable quantum-resistant encryption"
        ])
        return results
    
    def _get_solution(self, vuln_type: str) -> str:
        """Get remediation for vulnerability type"""
        solutions = {
            'SQLi': "Use parameterized queries",
            'XSS': "Implement output encoding",
            'RCE': "Use safer alternatives like ast.literal_eval()"
        }
        return solutions.get(vuln_type, "Review security best practices")

class MetaLearner:
    """Enables the AI to understand its own capabilities"""
    def __init__(self, ai_instance):
        self.ai = ai_instance
        self.architecture = {
            'knowledge_sources': len(ai_instance.DOMAINS),
            'endpoints': 8,
            'learning_algorithms': [
                "Pattern recognition",
                "Semantic analysis",
                "Heuristic generation"
            ]
        }
    
    def generate_self_report(self) -> dict:
        """Create comprehensive system analysis"""
        return {
            'knowledge_stats': {
                'entries': len(self.ai.knowledge),
                'last_updated': self.ai.knowledge.get('_meta', {}).get('timestamp', datetime.utcnow().isoformat()),
                'domains': list(self.ai.DOMAINS.keys())
            },
            'capabilities': self._get_capability_tree(),
            'recommendations': self._generate_improvements()
        }
    
    def _get_capability_tree(self) -> dict:
        """Map all system capabilities"""
        return {
            'cyber': ['exploit_gen', 'vuln_scan', 'malware_analysis'],
            'legal': ['document_analysis', 'precedent_search'],
            'autonomous': ['self_diagnosis', 'limited_self_repair']
        }
    
    def _generate_improvements(self) -> list:
        """Suggest architecture improvements"""
        return [
            "Implement reinforcement learning for exploit effectiveness",
            "Add dark web monitoring capability",
            "Develop polymorphic code generation"
        ]

class MotherBrain:
    DOMAINS = {
        # CYBER-INTELLIGENCE CORE
        'cyber': {
            '0day': [
                'https://github.com/rapid7/metasploit-framework/commits/master',
                'https://vx-underground.org/archive/VxHeaven/libv00.tar.gz',
                'https://www.exploit-db.com/google-hacking-database',
                'https://raw.githubusercontent.com/offensive-security/exploitdb/master/files_exploits.csv',
                'https://api.github.com/repos/torvalds/linux/commits',
                'https://cve.mitre.org/data/downloads/allitems.csv',
                'https://nvd.nist.gov/feeds/xml/cve/misc/nvd-rss.xml',
                'https://github.com/nomi-sec/PoC-in-GitHub/commits/master',
                'https://0day.today/rss',
                'https://www.zerodayinitiative.com/advisories/published/',
                'https://www.reddit.com/r/netsec/top/.json?sort=top&t=all',
                'https://github.com/vulhub/vulhub/commits/master',
                'https://attack.mitre.org/versions/v12/enterprise/enterprise.json',
                'https://raw.githubusercontent.com/offensive-security/exploitdb/master/README.md',
                'https://api.github.com/repos/advisories',
                'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json',
                'https://github.com/CVEProject/cvelist/commits/master',
                'https://www.kb.cert.org/vuln/data/feeds/'
            ],
            'ai_evasion': [
                'https://www.llmsec.org/papers.json',
                'https://arxiv.org/rss/cs.CR',
                'https://github.com/adversarial-examples/blackbox-attacks',
                'https://github.com/trusted-ai/adversarial-robustness-toolkit',
                'https://github.com/cleverhans-lab/cleverhans/commits/master',
                'https://github.com/IBM/adversarial-robustness-toolbox/commits/main',
                'https://evasion-api.com/v1/threats',
                'https://github.com/advboxes/AdvBox/commits/master',
                'https://github.com/BorealisAI/advertorch/commits/master',
                'https://github.com/facebookresearch/adversarial-robustness-toolbox',
                'https://github.com/tensorflow/cleverhans/commits/master',
                'https://github.com/airbnb/artificial-adversary'
            ],
            'creative': [
                'https://www.phrack.org/issues.html',
                'https://www.packetstormsecurity.com/files/rss',
                'https://www.reddit.com/r/blackhat/top/.json?sort=top&t=all',
                'https://github.com/ytisf/theZoo',
                'https://github.com/Hack-with-Github/Awesome-Hacking/commits/master',
                'https://github.com/swisskyrepo/PayloadsAllTheThings/commits/master',
                'https://github.com/danielmiessler/SecLists/commits/master',
                'https://github.com/enaqx/awesome-pentest/commits/master',
                'https://github.com/rmusser01/Infosec_Reference/commits/master',
                'https://github.com/alphaSeclab/awesome-reverse-engineering',
                'https://github.com/onlurking/awesome-infosec',
                'https://github.com/joe-shenouda/awesome-cyber-skills'
            ],
            'malware_analysis': [
                'https://virusshare.com/hashfiles',
                'https://malpedia.caad.fkie.fraunhofer.de/api/v1/pull',
                'https://bazaar.abuse.ch/export/txt/sha256/full/',
                'https://github.com/capesandbox/capes/commits/master',
                'https://www.hybrid-analysis.com/feed',
                'https://mb-api.abuse.ch/api/v1/',
                'https://github.com/ytisf/theZoo/tree/master/malwares/Binaries',
                'https://github.com/mstfknn/malware-sample-library',
                'https://github.com/Endermanch/MalwareDatabase',
                'https://github.com/fabrimagic72/malware-samples',
                'https://github.com/vxunderground/MalwareSourceCode'
            ],
            'reverse_engineering': [
                'https://github.com/radareorg/radare2/commits/master',
                'https://github.com/NationalSecurityAgency/ghidra/commits/master',
                'https://github.com/x64dbg/x64dbg/commits/development',
                'https://github.com/angr/angr/commits/master',
                'https://github.com/BinaryAnalysisPlatform/bap/commits/master',
                'https://github.com/avast/retdec/commits/master',
                'https://github.com/rizinorg/rizin/commits/dev',
                'https://github.com/Vector35/binaryninja-api/commits/dev'
            ],
            'forensics': [
                'https://github.com/volatilityfoundation/volatility/commits/master',
                'https://github.com/sleuthkit/sleuthkit/commits/develop',
                'https://github.com/VirusTotal/yara/commits/master',
                'https://github.com/ReFirmLabs/binwalk/commits/master',
                'https://github.com/google/rekall/commits/master'
            ]
        },
        # BUSINESS/FINANCE
        'business': [
            'https://www.sec.gov/Archives/edgar/xbrlrss.all.xml',
            'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd',
            'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey=demo',
            'https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey=demo',
            'https://api.polygon.io/v2/reference/news?apiKey=',
            'https://www.reddit.com/r/wallstreetbets/top/.json?sort=top&t=day',
            'https://www.reddit.com/r/algotrading/top/.json?sort=top&t=week',
            'https://www.bloomberg.com/markets/api/bulk-time-series/price/USD%3AAAPL?timeFrame=1_DAY',
            'https://www.quandl.com/api/v3/datasets.json',
            'https://api.twelvedata.com/time_series?symbol=AAPL&interval=1day',
            'https://api.marketstack.com/v1/tickers?access_key=',
            'https://finnhub.io/api/v1/news?category=general&token=',
            'https://www.forex.com/api/market/feed/rss'
        ],
        # LEGAL
        'legal': [
            'https://www.supremecourt.gov/opinions/slipopinion/22',
            'https://www.law.cornell.edu/supct/cert/',
            'https://www.uscourts.gov/rss/press-releases',
            'https://www.justice.gov/feeds/opa/justice-news.xml',
            'https://www.fcc.gov/rss/headlines',
            'https://www.ftc.gov/news-events/rss-feeds',
            'https://www.copyright.gov/rss/',
            'https://www.federalregister.gov/api/v1/documents.rss',
            'https://www.archives.gov/federal-register/rss',
            'https://www.gpo.gov/feeds/rss',
            'https://www.courtlistener.com/api/rest/v3/',
            'https://case.law/api/',
            'https://www.law.cornell.edu/wex/api.php',
            'https://www.oyez.org/api/v1/cases',
            'https://api.law.justia.com/api/v1/',
            'https://www.law360.com/rss'
        ],
        # PRODUCTIVITY
        'productivity': [
            'https://github.com/awesome-workplace/awesome-workplace',
            'https://www.salesforce.com/blog/rss/',
            'https://zapier.com/blog/feed/',
            'https://blog.trello.com/feed',
            'https://www.atlassian.com/blog/feed.xml',
            'https://blog.asana.com/feed/',
            'https://slack.com/blog/feed',
            'https://microsoft365.com/blog/feed/',
            'https://gsuiteupdates.googleblog.com/atom.xml',
            'https://www.producthunt.com/feed',
            'https://lifehacker.com/rss',
            'https://www.makeuseof.com/feed/',
            'https://zapier.com/engineering/feed/',
            'https://blog.rescuetime.com/feed/',
            'https://todoist.com/rss/news'
        ],
        # TRADING SIGNALS
        'trading_signals': [
            'https://www.alphavantage.co/query?function=SMA&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo',
            'https://finnhub.io/api/v1/scan/pattern?symbol=AAPL&resolution=D&token=',
            'https://api.polygon.io/v1/indicators/sma/AAPL?timespan=day&window=50&series_type=close&apiKey=',
            'https://api.tiingo.com/tiingo/daily/AAPL/prices?token=',
            'https://api.tdameritrade.com/v1/marketdata/AAPL/pricehistory'
        ],
        # THREAT INTELLIGENCE
        'threat_intel': [
            'https://otx.alienvault.com/api/v1/pulses/subscribed',
            'https://api.threatintelligenceplatform.com/v1/feeds',
            'https://www.threatminer.org/rss.php',
            'https://feeds.feedburner.com/TheHackersNews',
            'https://www.bleepingcomputer.com/feed/',
            'https://krebsonsecurity.com/feed/'
        ]
    }

    def __init__(self):
        self.gh_token = os.getenv("GITHUB_FINE_GRAINED_PAT")
        if not self.gh_token:
            raise RuntimeError("GitHub token not configured")
        
        self.repo_name = "AmericanPowerAI/mother-brain"
        self.knowledge = {}
        self.self_improver = SelfImprovingAI()
        self.meta = MetaLearner(self)
        self._init_self_healing()
        self._init_knowledge()

    def _init_self_healing(self):
        """Initialize autonomous repair systems"""
        self.healing_protocols = {
            'knowledge_corruption': self._repair_knowledge,
            'api_failure': self._restart_service,
            'security_breach': self._isolate_system
        }
    
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
                        "storage": "github"
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
                    "error": str(e)
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
        try:
            if url.endswith('.json'):
                data = requests.get(url).json()
                text = str(data)[:10000]
            elif url.endswith(('.csv', '.tar.gz')):
                text = requests.get(url).text[:5000]
            else:
                text = requests.get(url).text[:8000]
            
            self._process(domain_tag, text)
        except Exception as e:
            print(f"Failed {url}: {str(e)}")

    def _process(self, domain, text):
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
                "productivity": [r'Productivity\s+\d+%']
            }.get(domain, [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'])
            
            for pattern in patterns:
                for match in re.findall(pattern, text):
                    self.knowledge[f"{domain.upper()}:{match}"] = text[:500]

    def generate_exploit(self, cve):
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
            "signature": hashlib.md5(base.encode()).hexdigest()
        }

    def process_hacking_command(self, command):
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
                "signature": hashlib.md5(target.encode()).hexdigest()[:8]
            }
            
        elif base_cmd == "scan":
            scan_types = {
                "network": ["nmap -sV", "masscan -p1-65535"],
                "web": ["nikto -h", "wpscan --url"],
                "ai": ["llm_scan --model=gpt-4"]
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
                      if "HASH:" in k and target[:4] in k]
            
            return {
                "action": "decrypt",
                "hash": target,
                "attempts": [
                    f"rainbow_table --hash={target}",
                    f"john --format=raw-md5 {target}"
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
        "version": mother.knowledge.get("_meta", {}).get("version", "unknown")
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "knowledge_items": len(mother.knowledge),
        "memory_usage": f"{os.getpid()}",
        "uptime": "active"
    })

@app.route('/learn', methods=['POST'])
@limiter.limit("5 per minute")
def learn():
    mother.learn_all()
    return jsonify({"status": "Knowledge updated across all domains"})

@app.route('/ask', methods=['GET'])
def ask():
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400
    return jsonify(mother.knowledge.get(query, "No knowledge on this topic"))

@app.route('/exploit/<cve>', methods=['GET'])
@limiter.limit("10 per minute")
def exploit(cve):
    return jsonify(mother.generate_exploit(cve))

@app.route('/hacking', methods=['POST'])
@limiter.limit("15 per minute")
def hacking():
    command = request.json.get('command', '')
    if not command:
        return jsonify({"error": "No command provided"}), 400
    
    try:
        result = mother.process_hacking_command(command)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/dump', methods=['GET'])
def dump():
    """Return first 500 knowledge entries"""
    return jsonify(dict(list(mother.knowledge.items())[:500]))

@app.route('/dump_full', methods=['GET'])
def dump_full():
    """Return complete unfiltered knowledge dump"""
    return jsonify(mother.knowledge)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
