import logging
import json
import hashlib
from typing import Any, Optional, Callable, TypeVar, Union
from datetime import datetime, timedelta
from functools import wraps
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheEntry:
    """Represents a cached value with expiration."""
    
    def __init__(self, value: Any, ttl_seconds: int):
        self.value = value
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


class InMemoryCache:
    """Thread-safe in-memory cache implementation."""
    
    def __init__(self, default_ttl: int = 300):
        self._cache: dict = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._cache[key]
                return None
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in cache."""
        with self._lock:
            ttl = ttl or self.default_ttl
            self._cache[key] = CacheEntry(value, ttl)
    
    def delete(self, key: str):
        """Delete a value from cache."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self):
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self):
        """Remove all expired entries."""
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() 
                if v.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = len(self._cache)
            expired = sum(1 for v in self._cache.values() if v.is_expired)
            return {
                'total_entries': total,
                'active_entries': total - expired,
                'expired_entries': expired
            }


class RedisCache:
    """Redis-backed cache implementation."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = 'cryptopredictor:',
        default_ttl: int = 300
    ):
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._client = None
        
        try:
            import redis
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except ImportError:
            logger.warning("Redis package not installed, using in-memory cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}, using in-memory cache")
            self._client = None
    
    @property
    def is_available(self) -> bool:
        return self._client is not None
    
    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if not self._client:
            return None
        
        try:
            value = self._client.get(self._make_key(key))
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in cache."""
        if not self._client:
            return
        
        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value, default=str)
            self._client.setex(self._make_key(key), ttl, serialized)
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
    
    def delete(self, key: str):
        """Delete a value from cache."""
        if not self._client:
            return
        
        try:
            self._client.delete(self._make_key(key))
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
    
    def clear(self):
        """Clear all cached values with prefix."""
        if not self._client:
            return
        
        try:
            keys = self._client.keys(f"{self.prefix}*")
            if keys:
                self._client.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")


class HybridCache:
    """
    Hybrid cache using in-memory cache with optional Redis backing.
    Provides fast local access with Redis for persistence.
    """
    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
        local_ttl: int = 60,
        redis_ttl: int = 300
    ):
        self._local = InMemoryCache(default_ttl=local_ttl)
        self._redis: Optional[RedisCache] = None
        self.local_ttl = local_ttl
        self.redis_ttl = redis_ttl
        
        if redis_host:
            self._redis = RedisCache(
                host=redis_host,
                port=redis_port,
                default_ttl=redis_ttl
            )
    
    def get(self, key: str) -> Optional[Any]:
        """Get value, checking local cache first, then Redis."""
        # Check local cache first
        value = self._local.get(key)
        if value is not None:
            return value
        
        # Check Redis if available
        if self._redis and self._redis.is_available:
            value = self._redis.get(key)
            if value is not None:
                # Populate local cache
                self._local.set(key, value, self.local_ttl)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in both local and Redis cache."""
        self._local.set(key, value, ttl or self.local_ttl)
        
        if self._redis and self._redis.is_available:
            self._redis.set(key, value, ttl or self.redis_ttl)
    
    def delete(self, key: str):
        """Delete from both caches."""
        self._local.delete(key)
        if self._redis:
            self._redis.delete(key)
    
    def clear(self):
        """Clear both caches."""
        self._local.clear()
        if self._redis:
            self._redis.clear()


# Global cache instance
_cache: Optional[HybridCache] = None


def get_cache() -> HybridCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = HybridCache()
    return _cache


def init_cache(redis_host: Optional[str] = None, redis_port: int = 6379):
    """Initialize the global cache with optional Redis."""
    global _cache
    _cache = HybridCache(redis_host=redis_host, redis_port=redis_port)
    return _cache


def make_cache_key(*args, **kwargs) -> str:
    """Create a cache key from arguments."""
    key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(ttl: int = 300, key_prefix: str = ''):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds.
        key_prefix: Optional prefix for cache keys.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache = get_cache()
            
            # Generate cache key
            func_key = f"{key_prefix}{func.__module__}.{func.__name__}"
            arg_key = make_cache_key(*args, **kwargs)
            cache_key = f"{func_key}:{arg_key}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func_key}")
                return cached_value
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func_key}, cached result")
            
            return result
        return wrapper
    return decorator
