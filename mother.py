import json
import lzma
import re
import os
import requests
import random
import hashlib
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

class MotherBrain:
    DOMAINS = {
        # CYBER-INTELLIGENCE CORE
        'cyber': {
            '0day': [
                'https://github.com/rapid7/metasploit-framework/commits/master',
                'https://vx-underground.org/archive/VxHeaven/libv00.tar.gz',
                'https://www.exploit-db.com/google-hacking-database',
                'https://raw.githubusercontent.com/offensive-security/exploitdb/master/files_exploits.csv',
                'https://api.github.com/repos/torvalds/linux/commits'
            ],
            'ai_evasion': [
                'https://www.llmsec.org/papers.json',
                'https://arxiv.org/rss/cs.CR',
                'https://github.com/adversarial-examples/blackbox-attacks',
                'https://github.com/trusted-ai/adversarial-robustness-toolkit'
            ],
            'creative': [
                'https://www.phrack.org/issues.html',
                'https://www.packetstormsecurity.com/files/rss',
                'https://www.reddit.com/r/blackhat/top/.json?sort=top&t=all',
                'https://github.com/ytisf/theZoo'
            ],
            'malware_analysis': [
                'https://virusshare.com/hashfiles',
                'https://malpedia.caad.fkie.fraunhofer.de/api/v1/pull'
            ]
        },
        # BUSINESS/FINANCE
        'business': [
            'https://www.sec.gov/Archives/edgar/xbrlrss.all.xml',
            'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd'
        ],
        # LEGAL
        'legal': [
            'https://www.supremecourt.gov/opinions/slipopinion/22',
            'https://www.law.cornell.edu/supct/cert/'
        ],
        # PRODUCTIVITY
        'productivity': [
            'https://github.com/awesome-workplace/awesome-workplace',
            'https://www.salesforce.com/blog/rss/'
        ]
    }

    def __init__(self):
        self.gh_token = os.getenv("GITHUB_FINE_GRAINED_PAT")
        if not self.gh_token:
            raise RuntimeError("GitHub token not configured")
        
        self.repo_name = "AmericanPowerAI/mother-brain"
        self.knowledge = {}
        self._init_knowledge()

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
            "/health": "GET - System health check"
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
