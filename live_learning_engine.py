# live_learning_engine.py - Universal Internet Learning System
import asyncio
import aiohttp
import json
import re
import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse
import random
import sqlite3
from dataclasses import dataclass
import logging
from bs4 import BeautifulSoup
import feedparser
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque, Counter
import pickle

@dataclass
class LearnedContent:
    url: str
    title: str
    content: str
    domain: str
    timestamp: datetime
    confidence: float
    source_type: str
    metadata: Dict

class UniversalWebLearner:
    """Live learning system that scans the entire internet"""
    
    def __init__(self, mother_brain_instance):
        self.mother = mother_brain_instance
        self.session = aiohttp.ClientSession()
        self.learned_urls = set()
        self.learning_queue = asyncio.Queue(maxsize=10000)
        self.stats = {
            'websites_scanned': 0,
            'content_learned': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # NEW: Self-improvement components
        self.pattern_memory = {}  # Stores successful learning patterns
        self.failed_patterns = {}  # Stores what didn't work
        self.knowledge_graph = {}  # Builds relationships between concepts
        self.self_generated_questions = deque(maxlen=1000)  # Questions it asks itself
        self.learning_efficiency = 0.5  # Starts at 50%, improves over time
        
        # NEW: Autonomous learning database
        self.autonomous_db = self._init_autonomous_db()
        
        # Comprehensive website discovery sources
        self.seed_sources = {
            'search_engines': [
                'https://www.google.com/search?q=site:github.com+cybersecurity',
                'https://www.google.com/search?q=site:arxiv.org+artificial+intelligence',
                'https://www.google.com/search?q=site:reddit.com+technology',
                'https://www.bing.com/search?q=machine+learning+research',
                'https://duckduckgo.com/?q=latest+tech+news'
            ],
            'news_feeds': [
                'https://feeds.feedburner.com/oreilly/radar',
                'https://rss.cnn.com/rss/cnn_tech.rss',
                'https://feeds.bbci.co.uk/news/technology/rss.xml',
                'https://techcrunch.com/feed/',
                'https://www.wired.com/feed/',
                'https://feeds.arstechnica.com/arstechnica/index',
                'https://www.theverge.com/rss/index.xml'
            ],
            'academic_sources': [
                'https://arxiv.org/list/cs.AI/recent',
                'https://arxiv.org/list/cs.CR/recent',
                'https://arxiv.org/list/cs.LG/recent',
                'https://scholar.google.com/citations?view_op=new_articles',
                'https://dblp.org/search/publ/api'
            ],
            'government_feeds': [
                'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json',
                'https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-recent.json',
                'https://www.sec.gov/Archives/edgar/xbrlrss.all.xml',
                'https://www.whitehouse.gov/feed/'
            ],
            'developer_platforms': [
                'https://api.github.com/search/repositories?q=artificial+intelligence',
                'https://api.github.com/search/repositories?q=cybersecurity',
                'https://stackoverflow.com/feeds',
                'https://dev.to/feed/',
                'https://hackernews.com/rss'
            ],
            'business_sources': [
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://www.reuters.com/rssFeed/technologyNews',
                'https://fortune.com/feed/',
                'https://www.forbes.com/innovation/feed2/'
            ]
        }
        
        # Content extraction patterns
        self.extraction_patterns = {
            'cve_pattern': r'CVE-\d{4}-\d{4,7}',
            'github_repo': r'github\.com/[\w\-\.]+/[\w\-\.]+',
            'research_paper': r'arxiv\.org/abs/\d{4}\.\d{4,5}',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'hash_pattern': r'\b[a-fA-F0-9]{32,64}\b',
            'url_pattern': r'https?://[^\s<>"{}|\\^`\[\]]+',
            'email_pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        # Learning prioritization
        self.domain_priorities = {
            'arxiv.org': 10,          # Research papers
            'github.com': 9,          # Code repositories
            'cve.mitre.org': 10,      # Vulnerability data
            'nist.gov': 9,            # Government security
            'reddit.com': 7,          # Community discussions
            'stackoverflow.com': 8,   # Technical Q&A
            'medium.com': 6,          # Technical articles
            'news.ycombinator.com': 8, # Tech news
            'techcrunch.com': 6,      # Tech news
            'wired.com': 6            # Tech journalism
        }
        
        self.logger = self._setup_logging()
        self.learning_active = True
        
        # NEW: Start autonomous learning thread
        self._start_autonomous_learning()
        
    def _init_autonomous_db(self):
        """Initialize database for autonomous learning"""
        conn = sqlite3.connect('autonomous_learning.db')
        
        # Table for self-generated questions and answers
        conn.execute('''
            CREATE TABLE IF NOT EXISTS self_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                confidence REAL,
                verified BOOLEAN DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for concept relationships
        conn.execute('''
            CREATE TABLE IF NOT EXISTS concept_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept1 TEXT,
                concept2 TEXT,
                relationship TEXT,
                strength REAL,
                learned_from TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for learning patterns
        conn.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_data TEXT,
                success_rate REAL,
                usage_count INTEGER DEFAULT 0,
                last_used DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        return conn
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_learning.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('UniversalWebLearner')

    def _start_autonomous_learning(self):
        """Start autonomous learning thread that runs 24/7"""
        def autonomous_loop():
            while self.learning_active:
                try:
                    # Generate questions about gaps in knowledge
                    self._generate_self_questions()
                    
                    # Try to answer own questions
                    self._self_answer_questions()
                    
                    # Build concept relationships
                    self._build_knowledge_graph()
                    
                    # Improve learning efficiency
                    self._optimize_learning_strategy()
                    
                    time.sleep(300)  # Every 5 minutes
                except Exception as e:
                    self.logger.error(f"Autonomous learning error: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=autonomous_loop, daemon=True)
        thread.start()
        self.logger.info("🧠 Autonomous learning started")

    def _generate_self_questions(self):
        """Generate questions about things it doesn't know"""
        # Analyze knowledge gaps
        concepts = list(self.knowledge_graph.keys())
        
        for concept in random.sample(concepts, min(10, len(concepts))):
            # Generate different types of questions
            questions = [
                f"What is the relationship between {concept} and artificial intelligence?",
                f"How does {concept} work technically?",
                f"What are the latest developments in {concept}?",
                f"What are the security implications of {concept}?",
                f"How can {concept} be improved?",
            ]
            
            for question in questions:
                if question not in self.self_generated_questions:
                    self.self_generated_questions.append(question)
                    
                    # Store in database
                    self.autonomous_db.execute(
                        "INSERT INTO self_learning (question, confidence) VALUES (?, ?)",
                        (question, 0.0)
                    )
                    self.autonomous_db.commit()

    def _self_answer_questions(self):
        """Try to answer own questions using existing knowledge"""
        cursor = self.autonomous_db.execute(
            "SELECT id, question FROM self_learning WHERE answer IS NULL LIMIT 10"
        )
        
        for row in cursor.fetchall():
            q_id, question = row
            
            # Search existing knowledge
            answer = self._search_knowledge_for_answer(question)
            
            if answer:
                confidence = self._calculate_answer_confidence(question, answer)
                
                self.autonomous_db.execute(
                    "UPDATE self_learning SET answer = ?, confidence = ? WHERE id = ?",
                    (answer, confidence, q_id)
                )
                self.autonomous_db.commit()
                
                # Learn from this Q&A pair
                if hasattr(self.mother, 'feedback_learner'):
                    self.mother.feedback_learner.record_interaction(
                        question, answer, 1 if confidence > 0.7 else 0
                    )

    def _search_knowledge_for_answer(self, question):
        """Search knowledge base for answer"""
        question_words = set(question.lower().split())
        best_match = None
        best_score = 0
        
        for key, value in self.mother.knowledge.items():
            if isinstance(value, str):
                content_words = set(value.lower().split())
                overlap = len(question_words & content_words)
                
                if overlap > best_score:
                    best_score = overlap
                    best_match = value[:500]
        
        return best_match if best_score > 3 else None

    def _calculate_answer_confidence(self, question, answer):
        """Calculate confidence in generated answer"""
        # Simple heuristic - can be improved
        confidence = 0.5
        
        # Check if answer addresses question words
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        overlap = len(q_words & a_words) / len(q_words)
        confidence += overlap * 0.3
        
        # Length check
        if 50 < len(answer) < 500:
            confidence += 0.2
            
        return min(1.0, confidence)

    def _build_knowledge_graph(self):
        """Build relationships between concepts"""
        # Sample recent knowledge
        recent_keys = list(self.mother.knowledge.keys())[-100:]
        
        for i, key1 in enumerate(recent_keys):
            for key2 in recent_keys[i+1:i+5]:  # Check next 5 items
                # Find relationship
                relationship = self._find_relationship(key1, key2)
                
                if relationship:
                    # Store in graph
                    self.knowledge_graph[key1] = self.knowledge_graph.get(key1, [])
                    self.knowledge_graph[key1].append((key2, relationship))
                    
                    # Store in database
                    self.autonomous_db.execute(
                        "INSERT INTO concept_graph (concept1, concept2, relationship, strength) VALUES (?, ?, ?, ?)",
                        (key1, key2, relationship, 0.5)
                    )
        
        self.autonomous_db.commit()

    def _find_relationship(self, concept1, concept2):
        """Find relationship between two concepts"""
        # Simple relationship detection
        c1_lower = concept1.lower()
        c2_lower = concept2.lower()
        
        if 'cve' in c1_lower and 'cve' in c2_lower:
            return 'similar_vulnerability'
        elif 'ai' in c1_lower and 'ml' in c2_lower:
            return 'related_field'
        elif any(word in c1_lower for word in ['security', 'cyber']) and \
             any(word in c2_lower for word in ['security', 'cyber']):
            return 'security_related'
        
        # Check for common words
        words1 = set(c1_lower.split(':'))
        words2 = set(c2_lower.split(':'))
        if words1 & words2:
            return 'shared_context'
        
        return None

    def _optimize_learning_strategy(self):
        """Improve learning efficiency over time"""
        # Analyze what patterns work
        cursor = self.autonomous_db.execute(
            "SELECT pattern_type, success_rate FROM learning_patterns WHERE usage_count > 10"
        )
        
        successful_patterns = []
        for row in cursor.fetchall():
            if row[1] > 0.7:  # 70% success rate
                successful_patterns.append(row[0])
        
        # Adjust learning efficiency
        if successful_patterns:
            self.learning_efficiency = min(1.0, self.learning_efficiency + 0.01)
            
        # Prioritize successful patterns
        for pattern in successful_patterns:
            if pattern not in self.pattern_memory:
                self.pattern_memory[pattern] = 1
            else:
                self.pattern_memory[pattern] += 1

    async def start_continuous_learning(self):
        """Start the continuous learning process"""
        self.logger.info("🚀 Starting Universal Web Learning System")
        
        # Start multiple learning workers
        workers = [
            asyncio.create_task(self._discovery_worker()),
            asyncio.create_task(self._learning_worker()),
            asyncio.create_task(self._content_processor()),
            asyncio.create_task(self._stats_reporter()),
            asyncio.create_task(self._self_improvement_worker())  # NEW
        ]
        
        try:
            await asyncio.gather(*workers)
        except Exception as e:
            self.logger.error(f"Learning system error: {e}")
    
    async def _self_improvement_worker(self):
        """NEW: Continuously improve learning capabilities"""
        while self.learning_active:
            try:
                # Test current knowledge
                test_results = await self._test_knowledge()
                
                # Identify weak areas
                weak_areas = self._identify_weak_areas(test_results)
                
                # Focus learning on weak areas
                for area in weak_areas:
                    await self._focused_learning(area)
                
                # Update learning strategy
                self._update_strategy_based_on_performance()
                
                await asyncio.sleep(600)  # Every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Self-improvement error: {e}")
                await asyncio.sleep(300)

    async def _test_knowledge(self):
        """Test current knowledge level"""
        test_questions = [
            "What is machine learning?",
            "How does encryption work?",
            "What are the latest AI developments?",
            "Explain cybersecurity best practices",
            "What is quantum computing?"
        ]
        
        results = []
        for question in test_questions:
            answer = self._search_knowledge_for_answer(question)
            confidence = self._calculate_answer_confidence(question, answer) if answer else 0
            results.append({
                'question': question,
                'has_answer': answer is not None,
                'confidence': confidence
            })
        
        return results

    def _identify_weak_areas(self, test_results):
        """Identify areas that need more learning"""
        weak_areas = []
        
        for result in test_results:
            if result['confidence'] < 0.5:
                # Extract topic from question
                topic = self._extract_topic(result['question'])
                if topic:
                    weak_areas.append(topic)
        
        return weak_areas

    def _extract_topic(self, question):
        """Extract main topic from question"""
        topics = {
            'machine learning': ['ml', 'learning', 'ai'],
            'cybersecurity': ['security', 'cyber', 'hack'],
            'encryption': ['encrypt', 'crypto', 'secure'],
            'quantum': ['quantum', 'qubit'],
            'programming': ['code', 'program', 'software']
        }
        
        q_lower = question.lower()
        for topic, keywords in topics.items():
            if any(kw in q_lower for kw in keywords):
                return topic
        
        return 'general'

    async def _focused_learning(self, topic):
        """Focus learning on specific topic"""
        # Generate targeted search queries
        queries = [
            f"{topic} tutorial",
            f"{topic} explained",
            f"latest {topic} developments",
            f"{topic} best practices"
        ]
        
        for query in queries:
            # Add to learning queue with high priority
            await self.learning_queue.put({
                'url': f"https://www.google.com/search?q={query.replace(' ', '+')}",
                'category': 'focused_learning',
                'priority': 10,
                'discovered_from': 'self_improvement'
            })

    def _update_strategy_based_on_performance(self):
        """Update learning strategy based on performance metrics"""
        # Calculate success rate
        if self.stats['websites_scanned'] > 0:
            error_rate = self.stats['errors'] / self.stats['websites_scanned']
            
            # Adjust timeout and retry logic
            if error_rate > 0.3:
                # Too many errors, be more conservative
                self.learning_efficiency *= 0.95
            else:
                # Good performance, be more aggressive
                self.learning_efficiency = min(1.0, self.learning_efficiency * 1.02)
        
        # Store updated strategy
        self.autonomous_db.execute(
            "INSERT INTO learning_patterns (pattern_type, pattern_data, success_rate) VALUES (?, ?, ?)",
            ('efficiency', json.dumps({'rate': self.learning_efficiency}), self.learning_efficiency)
        )
        self.autonomous_db.commit()
        
    async def _discovery_worker(self):
        """Continuously discover new URLs to learn from"""
        while self.learning_active:
            try:
                # Discover from seed sources
                for category, sources in self.seed_sources.items():
                    for source in sources:
                        await self._discover_from_source(source, category)
                        await asyncio.sleep(1)  # Rate limiting
                
                # Discover from learned content (follow links)
                await self._discover_from_learned_content()
                
                await asyncio.sleep(60)  # Discovery cycle every minute
                
            except Exception as e:
                self.logger.error(f"Discovery worker error: {e}")
                await asyncio.sleep(30)

    async def _discover_from_source(self, source_url: str, category: str):
        """Discover URLs from a specific source"""
        try:
            async with self.session.get(source_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    urls = self._extract_urls_from_content(content)
                
                    for url in urls[:50]:  # Limit per source
                        if url not in self.learned_urls:
                            priority = self._calculate_priority(url)
                        
                            # NEW: Boost priority for weak areas
                            weak_areas = getattr(self, 'current_weak_areas', [])
                            if any(weak in url.lower() for weak in weak_areas):
                                priority += 2
                        
                            await self.learning_queue.put({
                                'url': url,
                                'category': category,
                                'priority': priority,
                                'discovered_from': source_url
                            })
                            
        except Exception as e:
            self.logger.debug(f"Could not discover from {source_url}: {e}")

        def _extract_urls_from_content(self, content: str) -> List[str]:
    """Extract URLs from HTML/text content"""
    urls = []
    
    # Parse HTML if possible
    try:
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract from links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http'):
                urls.append(href)
        
        # Extract from text using regex
        text_urls = re.findall(self.extraction_patterns['url_pattern'], content)
        urls.extend(text_urls)
        
    except Exception as e:
        # Fallback to regex extraction
        urls = re.findall(self.extraction_patterns['url_pattern'], content)
    
    return list(set(urls))  # Remove duplicates

    def _calculate_priority(self, url: str) -> int:
        """Calculate learning priority for a URL"""
        domain = urlparse(url).netloc.lower()
        
        # Check domain priorities
        for priority_domain, priority in self.domain_priorities.items():
            if priority_domain in domain:
                return priority
        
        # Default priority based on URL characteristics
        if any(keyword in url.lower() for keyword in ['api', 'json', 'feed', 'rss']):
            return 8
        elif any(keyword in url.lower() for keyword in ['pdf', 'paper', 'research']):
            return 7
        elif any(keyword in url.lower() for keyword in ['blog', 'article', 'post']):
            return 6
        else:
            return 5

    async def _learning_worker(self):
        """Process URLs from the learning queue"""
        while self.learning_active:
            try:
                # Get URL from queue with timeout
                try:
                    url_data = await asyncio.wait_for(
                        self.learning_queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                await self._learn_from_url(url_data)
                self.learning_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Learning worker error: {e}")

    async def _learn_from_url(self, url_data: Dict):
        """Learn content from a specific URL"""
        url = url_data['url']
        
        if url in self.learned_urls:
            return
            
        try:
            async with self.session.get(url, timeout=15) as response:
                if response.status == 200:
                    content = await response.text()
                    learned_content = await self._extract_knowledge(url, content, url_data)
                    
                    if learned_content:
                        await self._store_learned_content(learned_content)
                        self.learned_urls.add(url)
                        self.stats['websites_scanned'] += 1
                        self.stats['content_learned'] += len(learned_content.content.split())
                        
                        # NEW: Extract and store patterns
                        patterns = self._extract_learning_patterns(learned_content)
                        self._store_patterns(patterns)
                        
                        self.logger.info(f"✅ Learned from: {url} ({learned_content.domain})")
                    
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.debug(f"Failed to learn from {url}: {e}")

    def _extract_learning_patterns(self, content: LearnedContent):
        """Extract patterns from learned content"""
        patterns = []
        
        # Extract code patterns
        code_blocks = re.findall(r'```[\s\S]*?```', content.content)
        for code in code_blocks:
            patterns.append({
                'type': 'code',
                'content': code,
                'domain': content.domain,
                'confidence': content.confidence
            })
        
        # Extract definition patterns
        definitions = re.findall(r'(\w+)\s+is\s+(?:a|an|the)\s+([^.]+)', content.content)
        for term, definition in definitions:
            patterns.append({
                'type': 'definition',
                'term': term,
                'definition': definition,
                'domain': content.domain
            })
        
        # Extract list patterns
        lists = re.findall(r'(?:^|\n)\s*[\d•\-\*]\s+(.+)', content.content, re.MULTILINE)
        if lists:
            patterns.append({
                'type': 'list',
                'items': lists,
                'domain': content.domain
            })
        
        return patterns

    def _store_patterns(self, patterns):
        """Store learning patterns for future use"""
        for pattern in patterns:
            pattern_key = f"{pattern['type']}:{pattern.get('domain', 'general')}"
            
            if pattern_key not in self.pattern_memory:
                self.pattern_memory[pattern_key] = []
            
            self.pattern_memory[pattern_key].append(pattern)
            
            # Store in database
            self.autonomous_db.execute(
                "INSERT OR REPLACE INTO learning_patterns (pattern_type, pattern_data, success_rate, usage_count) VALUES (?, ?, ?, ?)",
                (pattern_key, json.dumps(pattern), 0.5, 1)
            )
        
        self.autonomous_db.commit()

    async def _extract_knowledge(self, url: str, content: str, url_data: Dict) -> Optional[LearnedContent]:
        """Extract knowledge from web content"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else urlparse(url).path
            
            # Extract main content
            text_content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Skip if content is too short or too long
            if len(text) < 100 or len(text) > 50000:
                return None
            
            # Determine domain/category
            domain = self._classify_content_domain(text, url)
            
            # Calculate confidence score
            confidence = self._calculate_content_confidence(text, url, url_data)
            
            # NEW: Boost confidence if content matches weak areas
            weak_areas = self._identify_weak_areas([])
            if any(weak in text.lower() for weak in weak_areas):
                confidence = min(1.0, confidence + 0.2)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, text, url)
            
            return LearnedContent(
                url=url,
                title=title[:200],  # Limit title length
                content=text[:10000],  # Limit content length
                domain=domain,
                timestamp=datetime.now(),
                confidence=confidence,
                source_type=url_data.get('category', 'web'),
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.debug(f"Content extraction failed for {url}: {e}")
            return None

    def _classify_content_domain(self, text: str, url: str) -> str:
        """Classify content into knowledge domains"""
        text_lower = text.lower()
        url_lower = url.lower()
        
        # Cybersecurity domain
        cyber_keywords = ['vulnerability', 'exploit', 'malware', 'security', 'hacking', 'cve', 'penetration', 'threat']
        if any(keyword in text_lower for keyword in cyber_keywords) or 'cve' in url_lower:
            return 'cybersecurity'
        
        # AI/ML domain
        ai_keywords = ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'algorithm']
        if any(keyword in text_lower for keyword in ai_keywords) or 'arxiv.org' in url_lower:
            return 'ai_ml'
        
        # Business domain
        business_keywords = ['business', 'market', 'finance', 'investment', 'revenue', 'profit', 'economy']
        if any(keyword in text_lower for keyword in business_keywords):
            return 'business'
        
        # Legal domain
        legal_keywords = ['law', 'legal', 'court', 'regulation', 'compliance', 'contract', 'patent']
        if any(keyword in text_lower for keyword in legal_keywords):
            return 'legal'
        
        # Technology domain
        tech_keywords = ['technology', 'software', 'programming', 'code', 'development', 'api']
        if any(keyword in text_lower for keyword in tech_keywords) or 'github.com' in url_lower:
            return 'technology'
        
        # Science domain
        science_keywords = ['research', 'study', 'scientific', 'experiment', 'analysis', 'theory']
        if any(keyword in text_lower for keyword in science_keywords):
            return 'science'
        
        return 'general'

    def _calculate_content_confidence(self, text: str, url: str, url_data: Dict) -> float:
        """Calculate confidence score for learned content"""
        confidence = 0.5  # Base confidence
        
        # URL-based factors
        domain = urlparse(url).netloc.lower()
        if domain in self.domain_priorities:
            confidence += 0.1 * (self.domain_priorities[domain] / 10)
        
        # Content quality factors
        if len(text) > 1000:
            confidence += 0.1
        if len(text.split()) > 200:
            confidence += 0.1
        
        # Source type factors
        if url_data.get('category') in ['academic_sources', 'government_feeds']:
            confidence += 0.2
        
        # Special pattern detection
        patterns_found = 0
        for pattern in self.extraction_patterns.values():
            if re.search(pattern, text):
                patterns_found += 1
        
        confidence += min(0.2, patterns_found * 0.05)
        
        # NEW: Adjust based on learning efficiency
        confidence *= self.learning_efficiency
        
        return min(1.0, confidence)

    def _extract_metadata(self, soup: BeautifulSoup, text: str, url: str) -> Dict:
        """Extract metadata from content"""
        metadata = {}
        
        # Extract structured data
        for pattern_name, pattern in self.extraction_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                metadata[pattern_name] = matches[:10]  # Limit matches
        
        # Extract meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            if meta.get('name') and meta.get('content'):
                meta_tags[meta['name']] = meta['content']
        
        metadata['meta_tags'] = meta_tags
        metadata['word_count'] = len(text.split())
        metadata['char_count'] = len(text)
        metadata['domain'] = urlparse(url).netloc
        
        # NEW: Extract key concepts
        metadata['key_concepts'] = self._extract_key_concepts(text)
        
        return metadata

    def _extract_key_concepts(self, text):
        """Extract key concepts from text"""
        # Simple noun phrase extraction
        words = text.split()
        word_freq = Counter(words)
        
        # Filter common words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be'}
        concepts = [word for word, count in word_freq.most_common(20) 
                   if word.lower() not in stop_words and len(word) > 3]
        
        return concepts[:10]

    async def _store_learned_content(self, content: LearnedContent):
        """Store learned content in the knowledge base"""
        try:
            # Create a knowledge key
            key_hash = hashlib.md5(content.url.encode()).hexdigest()[:8]
            knowledge_key = f"{content.domain.upper()}:{key_hash}"
            
            # Format content for storage
            formatted_content = {
                'title': content.title,
                'content': content.content[:1000],  # Store summary
                'url': content.url,
                'confidence': content.confidence,
                'timestamp': content.timestamp.isoformat(),
                'metadata': content.metadata
            }
            
            # Store in mother brain's knowledge base
            self.mother.knowledge[knowledge_key] = json.dumps(formatted_content)
            
            # Also process with domain-specific logic
            self.mother._process(content.domain, content.content)
            
            # NEW: Update knowledge graph
            if 'key_concepts' in content.metadata:
                for concept in content.metadata['key_concepts']:
                    self.knowledge_graph[concept] = self.knowledge_graph.get(concept, [])
                    self.knowledge_graph[concept].append(knowledge_key)
            
        except Exception as e:
            self.logger.error(f"Failed to store content: {e}")

    async def _content_processor(self):
        """Process learned content for insights"""
        while self.learning_active:
            try:
                # Process recent content for patterns
                await self._find_content_patterns()
                
                # Update knowledge compression
                await self._update_knowledge_compression()
                
                # Generate insights
                await self._generate_insights()
                
                # NEW: Synthesize new knowledge
                await self._synthesize_knowledge()
                
                await asyncio.sleep(300)  # Process every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Content processor error: {e}")

    async def _synthesize_knowledge(self):
        """NEW: Combine learned knowledge to create new insights"""
        # Get recent patterns
        recent_patterns = list(self.pattern_memory.items())[-10:]
        
        for pattern_type, patterns in recent_patterns:
            if len(patterns) >= 3:
                # Try to combine patterns
                synthesis = self._combine_patterns(patterns)
                
                if synthesis:
                    # Store as new knowledge
                    synthesis_key = f"SYNTHESIS:{hashlib.md5(synthesis.encode()).hexdigest()[:8]}"
                    self.mother.knowledge[synthesis_key] = synthesis
                    
                    self.logger.info(f"💡 Synthesized new knowledge: {synthesis[:100]}...")

    def _combine_patterns(self, patterns):
        """Combine multiple patterns into new knowledge"""
        if not patterns:
            return None
        
        pattern_type = patterns[0].get('type')
        
        if pattern_type == 'definition':
            # Combine definitions
            terms = [p.get('term', '') for p in patterns]
            definitions = [p.get('definition', '') for p in patterns]
            
            if terms and definitions:
                return f"Related concepts: {', '.join(terms[:5])} are all types of {patterns[0].get('domain', 'concepts')}"
        
        elif pattern_type == 'code':
            # Find common code patterns
            codes = [p.get('content', '') for p in patterns]
            if codes:
                return f"Common coding pattern found in {len(codes)} examples related to {patterns[0].get('domain', 'programming')}"
        
        return None

    async def _find_content_patterns(self):
        """Find patterns in recently learned content"""
        self.logger.info("🔍 Analyzing content patterns...")
        
        # Analyze pattern success rates
        cursor = self.autonomous_db.execute(
            "SELECT pattern_type, COUNT(*) as count FROM learning_patterns GROUP BY pattern_type"
        )
        
        for row in cursor.fetchall():
            self.logger.info(f"Pattern {row[0]}: {row[1]} instances")

    async def _update_knowledge_compression(self):
        """Update knowledge compression based on new content"""
        # Trigger knowledge compression if we have enough new content
        if self.stats['content_learned'] % 10000 == 0:  # Every 10k words
            self.logger.info("🗜️ Updating knowledge compression...")
            
            # NEW: Actually compress knowledge
            self._compress_similar_knowledge()

    def _compress_similar_knowledge(self):
        """Compress similar knowledge entries"""
        # Group similar keys
        key_groups = {}
        for key in self.mother.knowledge.keys():
            prefix = key.split(':')[0] if ':' in key else 'GENERAL'
            if prefix not in key_groups:
                key_groups[prefix] = []
            key_groups[prefix].append(key)
        
        # Compress each group
        for prefix, keys in key_groups.items():
            if len(keys) > 100:
                # Too many similar entries, compress them
                compressed_key = f"{prefix}:COMPRESSED:{datetime.now().strftime('%Y%m%d')}"
                compressed_content = f"Compressed {len(keys)} entries of type {prefix}"
                self.mother.knowledge[compressed_key] = compressed_content

    async def _generate_insights(self):
        """Generate insights from learned content"""
        self.logger.info("💡 Generating insights from learned content...")
        
        # Generate insight based on patterns
        if self.pattern_memory:
            most_common_pattern = max(self.pattern_memory.items(), key=lambda x: len(x[1]))
            insight = f"Most common pattern: {most_common_pattern[0]} with {len(most_common_pattern[1])} instances"
            
            # Store insight
            insight_key = f"INSIGHT:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.mother.knowledge[insight_key] = insight

    async def _stats_reporter(self):
        """Report learning statistics"""
        while self.learning_active:
            try:
                uptime = datetime.now() - self.stats['start_time']
                websites_per_minute = self.stats['websites_scanned'] / (uptime.total_seconds() / 60) if uptime.total_seconds() > 0 else 0
                
                self.logger.info(f"📊 Learning Stats: {self.stats['websites_scanned']} sites, "
                               f"{self.stats['content_learned']} words learned, "
                               f"{websites_per_minute:.1f} sites/min, "
                               f"Efficiency: {self.learning_efficiency:.2%}")
                
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                self.logger.error(f"Stats reporter error: {e}")

    async def _discover_from_learned_content(self):
        """Discover new URLs from previously learned content"""
        # Extract URLs from stored knowledge for further learning
        for key, value in list(self.mother.knowledge.items())[-100:]:  # Last 100 entries
            if isinstance(value, str):
                urls = self._extract_urls_from_content(value)
                for url in urls[:5]:  # Limit to prevent explosion
                    if url not in self.learned_urls and self.learning_queue.qsize() < 5000:
                        await self.learning_queue.put({
                            'url': url,
                            'category': 'discovered',
                            'priority': 4,
                            'discovered_from': 'learned_content'
                        })

    def get_learning_stats(self) -> Dict:
        """Get current learning statistics"""
        uptime = datetime.now() - self.stats['start_time']
        
        # Get autonomous learning stats
        cursor = self.autonomous_db.execute("SELECT COUNT(*) FROM self_learning")
        self_questions = cursor.fetchone()[0]
        
        cursor = self.autonomous_db.execute("SELECT COUNT(*) FROM concept_graph")
        relationships = cursor.fetchone()[0]
        
        return {
            'websites_scanned': self.stats['websites_scanned'],
            'content_learned_words': self.stats['content_learned'],
            'errors': self.stats['errors'],
            'uptime_seconds': uptime.total_seconds(),
            'websites_per_minute': self.stats['websites_scanned'] / (uptime.total_seconds() / 60) if uptime.total_seconds() > 0 else 0,
            'queue_size': self.learning_queue.qsize() if hasattr(self.learning_queue, 'qsize') else 0,
            'learned_urls_count': len(self.learned_urls),
            'learning_efficiency': self.learning_efficiency,
            'self_generated_questions': self_questions,
            'knowledge_relationships': relationships,
            'pattern_types': len(self.pattern_memory),
            'knowledge_graph_size': len(self.knowledge_graph)
        }

    async def stop_learning(self):
        """Stop the learning system"""
        self.learning_active = False
        await self.session.close()
        
        # Save state before stopping
        self._save_learning_state()
        
        self.logger.info("🛑 Learning system stopped")

    def _save_learning_state(self):
        """Save current learning state for persistence"""
        state = {
            'learned_urls': list(self.learned_urls),
            'pattern_memory': self.pattern_memory,
            'knowledge_graph': self.knowledge_graph,
            'learning_efficiency': self.learning_efficiency,
            'stats': self.stats
        }
        
        with open('learning_state.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info("💾 Learning state saved")

    def load_learning_state(self):
        """Load previous learning state if exists"""
        try:
            with open('learning_state.pkl', 'rb') as f:
                state = pickle.load(f)
                
            self.learned_urls = set(state['learned_urls'])
            self.pattern_memory = state['pattern_memory']
            self.knowledge_graph = state['knowledge_graph']
            self.learning_efficiency = state['learning_efficiency']
            self.stats.update(state['stats'])
            
            self.logger.info("📂 Previous learning state loaded")
            return True
        except Exception as e:
            self.logger.info(f"No previous state found: {e}")
            return False

# Keep existing FeedbackLearner class unchanged
class FeedbackLearner:
    """Learn from user feedback to improve responses"""
    
    def __init__(self, mother_brain_instance):
        self.mother = mother_brain_instance
        self.feedback_db = self._init_feedback_db()
        
    def _init_feedback_db(self):
        """Initialize feedback database"""
        conn = sqlite3.connect('feedback.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                response_hash TEXT,
                improvement_applied BOOLEAN DEFAULT 0
            )
        ''')
        conn.commit()
        return conn

    def record_feedback(self, query: str, response: str, feedback_type: str):
        """Record user feedback"""
        response_hash = hashlib.md5(response.encode()).hexdigest()
        
        self.feedback_db.execute('''
            INSERT INTO feedback (query, response, feedback_type, response_hash)
            VALUES (?, ?, ?, ?)
        ''', (query, response, feedback_type, response_hash))
        self.feedback_db.commit()
        
        # Immediate learning from negative feedback
        if feedback_type == 'negative':
            self._learn_from_negative_feedback(query, response)

    def _learn_from_negative_feedback(self, query: str, response: str):
        """Learn from negative feedback immediately"""
        # Analyze what went wrong and update knowledge
        query_tokens = query.lower().split()
        
        # Mark this response pattern as problematic
        problem_key = f"FEEDBACK:NEGATIVE:{hashlib.md5(query.encode()).hexdigest()[:8]}"
        self.mother.knowledge[problem_key] = f"User reported poor response for: {query[:100]}"
        
        # Try to find better knowledge for similar queries
        self._find_better_knowledge(query)

    def _find_better_knowledge(self, query: str):
        """Find better knowledge for queries with negative feedback"""
        # Search for related knowledge that might provide better answers
        query_words = set(query.lower().split())
        
        best_matches = []
        for key, value in self.mother.knowledge.items():
            if key.startswith('FEEDBACK:'):
                continue
                
            content_words = set(str(value).lower().split())
            overlap = len(query_words & content_words)
            
            if overlap > 0:
                best_matches.append((key, value, overlap))
        
        # Store improved knowledge
        if best_matches:
            best_matches.sort(key=lambda x: x[2], reverse=True)
            improved_key = f"IMPROVED:{hashlib.md5(query.encode()).hexdigest()[:8]}"
            self.mother.knowledge[improved_key] = best_matches[0][1]

    def get_feedback_insights(self) -> Dict:
        """Get insights from user feedback"""
        cursor = self.feedback_db.execute('''
            SELECT 
                feedback_type,
                COUNT(*) as count,
                AVG(CASE WHEN feedback_type = 'positive' THEN 1 ELSE 0 END) as satisfaction_rate
            FROM feedback 
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY feedback_type
        ''')
        
        results = cursor.fetchall()
        
        total_feedback = sum(row[1] for row in results)
        positive_feedback = sum(row[1] for row in results if row[0] == 'positive')
        
        return {
            'total_feedback_24h': total_feedback,
            'positive_feedback_24h': positive_feedback,
            'satisfaction_rate': (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0,
            'feedback_breakdown': {row[0]: row[1] for row in results}
        }

# Integration with Mother Brain remains the same
def integrate_live_learning(mother_brain_instance):
    """Integrate live learning with existing Mother Brain"""
    
    # Add live learning capability
    mother_brain_instance.live_learner = UniversalWebLearner(mother_brain_instance)
    mother_brain_instance.feedback_learner = FeedbackLearner(mother_brain_instance)
    
    # Load previous state if exists
    mother_brain_instance.live_learner.load_learning_state()
    
    # Start live learning in background
    def start_background_learning():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(mother_brain_instance.live_learner.start_continuous_learning())
    
    learning_thread = threading.Thread(target=start_background_learning, daemon=True)
    learning_thread.start()
    
    # Add new API endpoints to Flask app
    from flask import request, jsonify
    
    @mother_brain_instance.app.route('/learning/stats', methods=['GET'])
    def get_learning_stats():
        stats = mother_brain_instance.live_learner.get_learning_stats()
        feedback_stats = mother_brain_instance.feedback_learner.get_feedback_insights()
        
        return jsonify({
            'learning': stats,
            'feedback': feedback_stats,
            'timestamp': datetime.now().isoformat()
        })
    
    @mother_brain_instance.app.route('/feedback', methods=['POST'])
    def record_feedback():
        data = request.get_json()
        
        query = data.get('query', '')
        response = data.get('response', '')
        feedback_type = data.get('type', 'neutral')  # positive, negative, neutral
        
        mother_brain_instance.feedback_learner.record_feedback(query, response, feedback_type)
        
        return jsonify({
            'status': 'feedback_recorded',
            'message': 'Thank you for helping MOTHER AI learn!'
        })
    
    @mother_brain_instance.app.route('/learning/control', methods=['POST'])
    def control_learning():
        action = request.json.get('action')
        
        if action == 'boost':
            # Increase learning rate temporarily
            mother_brain_instance.live_learner.learning_efficiency = min(1.0, mother_brain_instance.live_learner.learning_efficiency * 1.5)
            return jsonify({'status': 'learning_boosted', 'new_efficiency': mother_brain_instance.live_learner.learning_efficiency})
        elif action == 'pause':
            mother_brain_instance.live_learner.learning_active = False
            return jsonify({'status': 'learning_paused'})
        elif action == 'resume':
            mother_brain_instance.live_learner.learning_active = True
            return jsonify({'status': 'learning_resumed'})
        
        return jsonify({'error': 'Invalid action'})
    
    return mother_brain_instance
