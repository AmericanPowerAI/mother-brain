# cache.py - Add this new file
import redis
import json
import hashlib
from typing import Any, Optional, Dict
from functools import wraps
import time

class MotherCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()  # Test connection
            self.enabled = True
            print("Cache initialized successfully")
        except Exception as e:
            print(f"Cache unavailable: {e}")
            self.enabled = False
            self.redis_client = None
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key"""
        key_data = f"{prefix}:{':'.join(map(str, args))}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value.decode('utf-8'))
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL (seconds)"""
        if not self.enabled:
            return False
        
        try:
            serialized = json.dumps(value, default=str)
            self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str):
        """Delete key from cache"""
        if not self.enabled:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str):
        """Delete all keys matching pattern"""
        if not self.enabled:
            return False
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            print(f"Cache invalidate error: {e}")
            return False

# Decorator for caching function results
def cached(cache_instance: MotherCache, ttl: int = 3600, prefix: str = ""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_instance._generate_key(
                prefix or func.__name__, *args, **kwargs
            )
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            return result
        
        # Add cache management methods
        wrapper.invalidate = lambda *args, **kwargs: cache_instance.delete(
            cache_instance._generate_key(prefix or func.__name__, *args, **kwargs)
        )
        wrapper.invalidate_all = lambda: cache_instance.invalidate_pattern(
            f"{prefix or func.__name__}:*"
        )
        
        return wrapper
    return decorator

# Update mother.py to use caching
class MotherBrainWithCache(MotherBrain):
    def __init__(self):
        super().__init__()
        self.cache = MotherCache()
    
    @cached(cache_instance=None, ttl=1800, prefix="knowledge")  # 30 min cache
    def get_knowledge(self, key: str) -> str:
        """Cached knowledge retrieval"""
        return super().get_knowledge(key)
    
    @cached(cache_instance=None, ttl=3600, prefix="exploit")  # 1 hour cache
    def generate_exploit(self, cve: str):
        """Cached exploit generation"""
        return super().generate_exploit(cve)
    
    @cached(cache_instance=None, ttl=300, prefix="search")  # 5 min cache
    def search_knowledge(self, query: str, domain: str = None):
        """Cached knowledge search"""
        if hasattr(self, 'db'):
            return self.db.search_knowledge(query, domain)
        else:
            # Fallback to original search
            results = []
            for key, value in self.knowledge.items():
                if query.lower() in key.lower() or query.lower() in str(value).lower():
                    if not domain or domain.lower() in key.lower():
                        results.append({"key": key, "value": str(value)[:200]})
                        if len(results) >= 10:
                            break
            return results
    
    def invalidate_knowledge_cache(self):
        """Invalidate all knowledge caches after updates"""
        if hasattr(self.get_knowledge, 'invalidate_all'):
            self.get_knowledge.invalidate_all()
        if hasattr(self.search_knowledge, 'invalidate_all'):
            self.search_knowledge.invalidate_all()
    
    def learn_all(self):
        """Override to invalidate cache after learning"""
        result = super().learn_all()
        self.invalidate_knowledge_cache()
        return result

# Cache warming strategies
class CacheWarmer:
    def __init__(self, mother_brain: MotherBrainWithCache):
        self.mother = mother_brain
    
    def warm_popular_queries(self):
        """Pre-load frequently accessed data"""
        popular_queries = [
            "CVE-2023",
            "exploit",
            "vulnerability",
            "malware",
            "0day"
        ]
        
        for query in popular_queries:
            try:
                self.mother.search_knowledge(query)
                time.sleep(0.1)  # Rate limit
            except Exception as e:
                print(f"Cache warming failed for {query}: {e}")
    
    def warm_recent_cves(self):
        """Pre-load recent CVE data"""
        import datetime
        current_year = datetime.datetime.now().year
        
        for year in [current_year, current_year - 1]:
            for month in range(1, 13):
                cve_pattern = f"CVE-{year}-{month:02d}"
                try:
                    self.mother.search_knowledge(cve_pattern, "cyber")
                    time.sleep(0.1)
                except Exception as e:
                    print(f"CVE cache warming failed: {e}")

# Performance monitoring for cache
class CacheMetrics:
    def __init__(self, cache: MotherCache):
        self.cache = cache
        self.hits = 0
        self.misses = 0
        self.start_time = time.time()
    
    def record_hit(self):
        self.hits += 1
    
    def record_miss(self):
        self.misses += 1
    
    def get_stats(self) -> Dict:
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        uptime = time.time() - self.start_time
        
        return {
            "hit_rate": f"{hit_rate:.2f}%",
            "total_hits": self.hits,
            "total_misses": self.misses,
            "uptime_seconds": uptime,
            "cache_enabled": self.cache.enabled
        }

# Add to requirements.txt:
# redis>=4.5.0
