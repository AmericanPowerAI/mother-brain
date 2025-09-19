# cache.py - Standalone caching module (no inheritance required)
import redis
import json
import hashlib
from typing import Any, Optional, Dict
from functools import wraps
import time

class MotherCache:
    """Standalone Redis cache manager"""
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

# Helper function to add caching to existing methods without modifying the class
def add_cache_to_method(obj, method_name: str, cache: MotherCache, ttl: int = 3600):
    """
    Dynamically add caching to an existing method without modifying the class.
    Usage: add_cache_to_method(mother, 'get_knowledge', mother.cache)
    """
    if not hasattr(obj, method_name):
        return
    
    original_method = getattr(obj, method_name)
    
    @wraps(original_method)
    def cached_method(*args, **kwargs):
        cache_key = cache._generate_key(method_name, *args, **kwargs)
        
        # Try cache first
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Call original method
        result = original_method(*args, **kwargs)
        
        # Cache the result
        cache.set(cache_key, result, ttl)
        return result
    
    # Replace the method with cached version
    setattr(obj, method_name, cached_method)

# Utility class to create cached versions of functions
class CacheWrapper:
    """Wrapper to add caching to any callable without modifying original code"""
    
    def __init__(self, cache: MotherCache):
        self.cache = cache
        self.metrics = CacheMetrics(cache)
    
    def wrap(self, func, ttl: int = 3600, prefix: str = None):
        """Wrap any function with caching"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = self.cache._generate_key(prefix or func.__name__, *args, **kwargs)
            
            # Check cache
            result = self.cache.get(cache_key)
            if result is not None:
                self.metrics.record_hit()
                return result
            
            # Cache miss
            self.metrics.record_miss()
            result = func(*args, **kwargs)
            self.cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    
    def cache_method(self, obj, method_name: str, ttl: int = 3600):
        """Cache a specific method of an object"""
        if hasattr(obj, method_name):
            original = getattr(obj, method_name)
            cached_version = self.wrap(original, ttl, f"{obj.__class__.__name__}.{method_name}")
            setattr(obj, method_name, cached_version)

# Cache warming strategies (works with any MotherBrain instance)
class CacheWarmer:
    def __init__(self, mother_brain):
        """Initialize with any MotherBrain instance"""
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
                if hasattr(self.mother, 'search_knowledge'):
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
                    if hasattr(self.mother, 'search_knowledge'):
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

# Factory function to create cached MotherBrain without inheritance
def create_cached_brain(mother_brain_instance):
    """
    Add caching to an existing MotherBrain instance without modifying it.
    This function can be called from mother.py after creating the instance.
    """
    if not hasattr(mother_brain_instance, 'cache'):
        mother_brain_instance.cache = MotherCache()
    
    wrapper = CacheWrapper(mother_brain_instance.cache)
    
    # Add caching to specific methods if they exist
    methods_to_cache = [
        ('get_knowledge', 1800),  # 30 min
        ('generate_exploit', 3600),  # 1 hour
        ('search_knowledge', 300),  # 5 min
    ]
    
    for method_name, ttl in methods_to_cache:
        wrapper.cache_method(mother_brain_instance, method_name, ttl)
    
    return mother_brain_instance

# Standalone cache manager for use without modifying MotherBrain
class StandaloneCacheManager:
    """
    Manages caching independently of MotherBrain class.
    Can be used to cache any function or method results.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.cache = MotherCache(redis_url)
        self.wrapper = CacheWrapper(self.cache)
        self.metrics = self.wrapper.metrics
    
    def cache_function(self, func, ttl: int = 3600):
        """Return a cached version of any function"""
        return self.wrapper.wrap(func, ttl)
    
    def get_cached_value(self, key: str):
        """Get value from cache by key"""
        return self.cache.get(key)
    
    def set_cached_value(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        return self.cache.set(key, value, ttl)
    
    def clear_cache(self, pattern: str = "*"):
        """Clear cache entries matching pattern"""
        return self.cache.invalidate_pattern(pattern)
    
    def get_metrics(self):
        """Get cache performance metrics"""
        return self.metrics.get_stats()

# Add to requirements.txt:
# redis>=4.5.0
