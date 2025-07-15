# conversation.py
import re
import random
import hashlib
import threading
from datetime import datetime
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Callable

# ======================
# CORE DATA STRUCTURES
# ======================
class Tone(Enum):
    PROFESSIONAL = 1
    FRIENDLY = 2
    TECHNICAL = 3
    GUARDIAN = 4

class Domain(Enum):
    TECH = "technology"
    HEALTH = "health"
    FINANCE = "finance"
    LEGAL = "legal"
    GENERAL = "general"

# ======================
# SECURITY LAYER
# ======================
class ThreatAnalyzer:
    BLACKLIST = [
        r'(\bpassword\b|\bsecret\b)[=:]\S+',
        r'<\s*script[^>]*>.*<\s*/\s*script\s*>',
        r'(?:%[0-9a-f]{2}|\\x[0-9a-f]{2})+'
    ]
    
    @staticmethod
    def analyze(text: str) -> int:
        return sum(
            1 for pattern in ThreatAnalyzer.BLACKLIST 
            if re.search(pattern, text, re.I)
        )

# ======================
# MULTI-DOMAIN BRAIN
# ======================
class OmniExpert:
    DOMAIN_KEYWORDS = {
        Domain.HEALTH: ["symptom", "diagnos", "treatment"],
        Domain.FINANCE: ["invest", "stock", "ROI"],
        Domain.LEGAL: ["contract", "liability", "clause"]
    }
    
    @classmethod
    def detect_domain(cls, text: str) -> Domain:
        text_lower = text.lower()
        for domain, keywords in cls.DOMAIN_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return domain
        return Domain.GENERAL

# ======================
# CONVERSATION ENGINE
# ======================
class ConversationCore:
    def __init__(self, max_history=100):
        self.memory = deque(maxlen=max_history)
        self.persona = PersonalityEngine()
        self.threat = ThreatAnalyzer()
        self.expert = OmniExpert()
        
    def respond(self, user_input: str) -> str:
        # Security check
        if self.threat.analyze(user_input) >= 3:
            return self._security_response()
            
        # Domain detection
        domain = self.expert.detect_domain(user_input)
        response = self._generate_response(user_input, domain)
        
        # Memory and enhancements
        self.memory.append((user_input, response))
        return self._enhance_response(response, domain)

    def _generate_response(self, query: str, domain: Domain) -> str:
        """Integrated with prompts.py templates"""
        template = {
            Domain.HEALTH: "medical_response",
            Domain.FINANCE: "financial_analysis",
            Domain.LEGAL: "legal_review"
        }.get(domain, "default_response")
        
        return f"[[{template}]] {query[:50]}..."  # Actual integration shown below

    def _enhance_response(self, response: str, domain: Domain) -> str:
        """Adds features like tips, challenges, etc."""
        enhancements = [
            self._add_domain_badge(domain),
            self._inject_tip(),
            self._daily_challenge()
        ]
        return response + "\n\n" + "\n".join(filter(None, enhancements))

# ======================
# USER-FACING FEATURES
# ======================
class PersonalityEngine:
    TONES = {
        Tone.GUARDIAN: {
            'opening': ["🔒 Security scan complete...", "⚠️ Caution advised..."],
            'closing': ["Stay shielded.", "Enable /guardian-mode for details"]
        },
        Tone.FRIENDLY: {
            'opening': ["Hey there!", "Interesting question!"],
            'closing': ["What else can I help with?", "Want me to simplify this?"]
        }
    }
    
    def apply_tone(self, text: str, tone: Tone) -> str:
        style = self.TONES.get(tone, self.TONES[Tone.FRIENDLY])
        return f"{random.choice(style['opening'])} {text} {random.choice(style['closing'])}"

class AwarenessFeatures:
    TIPS = [
        "💡 Try '/explain like I'm 5' for simple breakdowns",
        "💡 '/cite' adds academic references",
        "💡 Use '/privacy-mode' for self-destructing chats"
    ]
    
    CHALLENGES = [
        "🎯 StumpTheAI: Ask something obscure!",
        "🎯 5MinuteExpert: Master any topic fast"
    ]
    
    @staticmethod
    def inject_tip() -> Optional[str]:
        return random.choice(AwarenessFeatures.TIPS) if random.random() > 0.7 else None
        
    @staticmethod
    def daily_challenge() -> Optional[str]:
        return random.choice(AwarenessFeatures.CHALLENGES) if random.random() > 0.9 else None

# ======================
# ENTERPRISE FEATURES
# ======================
class EnterpriseModule:
    @staticmethod
    def compliance_log(query: str, response: str) -> Dict:
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'query_hash': hashlib.sha256(query.encode()).hexdigest(),
            'frameworks': ['GDPR'] if 'personal' in query.lower() else []
        }
    
    @staticmethod
    def generate_roi(use_cases: List[str]) -> str:
        savings = {"legal": 150, "research": 200}
        total = sum(savings.get(uc, 0) * 2080 for uc in use_cases)
        return f"Estimated annual savings: ${total:,}"

# ======================
# INTEGRATION LAYER
# ======================
class MotherBrainConversation:
    def __init__(self):
        self.core = ConversationCore()
        self.enterprise = EnterpriseModule()
        self.awareness = AwarenessFeatures()
        
    def chat(self, user_input: str, is_enterprise: bool = False) -> str:
        response = self.core.respond(user_input)
        
        if is_enterprise:
            self.enterprise.compliance_log(user_input, response)
            
        return response + self._get_enhancements(user_input)
    
    def _get_enhancements(self, query: str) -> str:
        return "\n".join(filter(None, [
            self.awareness.inject_tip(),
            self.awareness.daily_challenge(),
            "💼 Try /roi-calculator" if "cost" in query.lower() else None
        ]))

# ============================================
# PROMPTS.PY vs CONVERSATION.PY RELATIONSHIP
# ============================================
"""
Key Differences:
1. prompts.py - Contains:
   - Raw template strings
   - Formatting rules
   - Placeholder logic
   Example: "Answer {query} as a {expert_type} with {tone} tone"

2. conversation.py - Contains:
   - Dialogue flow control
   - Memory management
   - Feature orchestration
   Example: "When user asks X, select Y template from prompts.py, then apply Z tone"

Integration Example:
"""
# Sample prompts.py integration
import prompts  # Separate file

class UnifiedResponseBuilder:
    def __init__(self):
        self.templates = prompts.PROMPT_TEMPLATES
        
    def build_response(self, query: str, domain: str) -> str:
        template = self.templates.get(
            f"{domain}_response",
            self.templates["default_response"]
        )
        return template.format(
            query=query,
            expert_type=domain.capitalize(),
            tone="professional"
        )

# ======================
# QUICKSTART
# ======================
if __name__ == "__main__":
    mb = MotherBrainConversation()
    print(mb.chat("How do I secure my Linux server?"))
    print(mb.chat("Explain quantum computing", is_enterprise=True))
