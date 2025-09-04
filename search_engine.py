from duckduckgo_search import DDGS
import asyncio
from typing import List, Dict
import hashlib

class IntelligentSearchEngine:
    def __init__(self):
        self.ddgs = DDGS()
        self.search_history = {}
        
    async def search_and_verify(self, query: str, num_sources: int = 5) -> Dict:
        """Search multiple sources and cross-verify information"""
        
        # Search DuckDuckGo
        results = self.ddgs.text(query, max_results=num_sources)
        
        # Extract and analyze results
        verified_info = self.cross_reference_sources(results)
        
        # Cache results
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.search_history[query_hash] = verified_info
        
        return verified_info
    
    def cross_reference_sources(self, results: List) -> Dict:
        """Use logic to verify information across sources"""
        facts = {}
        confidence_scores = {}
        
        for result in results:
            # Extract facts from each source
            content = result.get('body', '')
            # Simple fact extraction (enhance with NLP)
            sentences = content.split('.')
            
            for sentence in sentences:
                # Count how many sources mention similar facts
                fact_hash = hashlib.md5(sentence.strip().encode()).hexdigest()
                if fact_hash in facts:
                    facts[fact_hash]['count'] += 1
                    facts[fact_hash]['sources'].append(result['href'])
                else:
                    facts[fact_hash] = {
                        'text': sentence.strip(),
                        'count': 1,
                        'sources': [result['href']]
                    }
        
        # Calculate confidence based on source agreement
        for fact_hash, fact_data in facts.items():
            confidence = fact_data['count'] / len(results)
            confidence_scores[fact_hash] = confidence
            
        return {
            'facts': facts,
            'confidence': confidence_scores,
            'sources': results
        }
