# database.py - Add this new file
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import threading
from contextlib import contextmanager

class KnowledgeDB:
    def __init__(self, db_path: str = "knowledge.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_tables()
    
    def _init_tables(self):
        """Create database schema"""
        with self.get_connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS cve_exploits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cve_id TEXT UNIQUE NOT NULL,
                    payload TEXT NOT NULL,
                    severity REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS user_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT,
                    user_ip TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge(domain);
                CREATE INDEX IF NOT EXISTS idx_knowledge_key ON knowledge(key);
                CREATE INDEX IF NOT EXISTS idx_cve_id ON cve_exploits(cve_id);
            ''')
    
    @contextmanager
    def get_connection(self):
        """Thread-safe database connection"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Dict-like access
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
    
    def store_knowledge(self, key: str, value: str, domain: str):
        """Store knowledge with domain classification"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO knowledge (key, value, domain, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (key, value, domain))
    
    def get_knowledge(self, key: str) -> Optional[str]:
        """Retrieve knowledge by key"""
        with self.get_connection() as conn:
            result = conn.execute(
                'SELECT value FROM knowledge WHERE key = ?', (key,)
            ).fetchone()
            return result['value'] if result else None
    
    def search_knowledge(self, query: str, domain: str = None) -> List[Dict]:
        """Search knowledge with optional domain filter"""
        with self.get_connection() as conn:
            if domain:
                results = conn.execute('''
                    SELECT key, value, domain FROM knowledge 
                    WHERE (key LIKE ? OR value LIKE ?) AND domain = ?
                    ORDER BY updated_at DESC LIMIT 20
                ''', (f'%{query}%', f'%{query}%', domain)).fetchall()
            else:
                results = conn.execute('''
                    SELECT key, value, domain FROM knowledge 
                    WHERE key LIKE ? OR value LIKE ?
                    ORDER BY updated_at DESC LIMIT 20
                ''', (f'%{query}%', f'%{query}%')).fetchall()
            
            return [dict(row) for row in results]
    
    def store_cve(self, cve_id: str, payload: str, severity: float):
        """Store CVE exploit data"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO cve_exploits (cve_id, payload, severity)
                VALUES (?, ?, ?)
            ''', (cve_id, payload, severity))
    
    def log_query(self, query: str, response: str, user_ip: str):
        """Log user queries for analytics"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO user_queries (query, response, user_ip)
                VALUES (?, ?, ?)
            ''', (query, response, user_ip))
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            stats = {}
            stats['total_knowledge'] = conn.execute(
                'SELECT COUNT(*) as count FROM knowledge'
            ).fetchone()['count']
            
            stats['domains'] = conn.execute('''
                SELECT domain, COUNT(*) as count 
                FROM knowledge 
                GROUP BY domain
            ''').fetchall()
            
            stats['recent_queries'] = conn.execute('''
                SELECT COUNT(*) as count 
                FROM user_queries 
                WHERE timestamp > datetime('now', '-24 hours')
            ''').fetchone()['count']
            
            return stats

def create_mother_brain_with_db():
    from mother import MotherBrain  # Import inside function to avoid circular import
    
    class MotherBrainWithDB(MotherBrain):
        def __init__(self):
            super().__init__()
            self.db = KnowledgeDB()
            self._migrate_github_to_db()
        
        def _migrate_github_to_db(self):
            """One-time migration from GitHub storage to database"""
            for key, value in self.knowledge.items():
                if not key.startswith('_'):
                    domain = key.split(':')[0] if ':' in key else 'general'
                    self.db.store_knowledge(key, str(value), domain)
        
        def _process(self, domain, text):
            """Override to use database storage"""
            # ... existing processing logic ...
            # Replace self.knowledge[key] = value with:
            self.db.store_knowledge(key, value, domain)
        
        def get_knowledge(self, key: str) -> str:
            """Get knowledge from database"""
            result = self.db.get_knowledge(key)
            return result if result else "No knowledge on this topic"
    
    return MotherBrainWithDB


