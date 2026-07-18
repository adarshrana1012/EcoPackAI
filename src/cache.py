"""
cache.py — Redis Caching Layer for EcoPackAI (Phase 8, Prompt P10)
==================================================================

Provides async Redis caching for classification inference results,
cache invalidation on model promotion, a health-check helper, and a
token-bucket rate limiter for all /v1/* gateway routes.

Cache Key Strategy
------------------
    key = sha256(json.dumps(product_features, sort_keys=True))

This makes cache hits deterministic across callers with identical
feature vectors, regardless of key ordering.

Usage
-----
    from src.cache import (
        get_cached_classification,
        set_cached_classification,
        invalidate_classification_cache,
        check_rate_limit,
        is_redis_connected,
    )

Author: EcoPackAI Team
"""

from __future__ import annotations

import hashlib
import json
import structlog
from typing import Any, Dict, Optional

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional: redis.asyncio
# ---------------------------------------------------------------------------
try:
    import redis.asyncio as aioredis  # type: ignore[import-untyped]
    _HAS_REDIS = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    _HAS_REDIS = False
    logger.warning("redis[asyncio] not installed. Caching and rate limiting are disabled.")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
from src.settings import get_settings

_settings = get_settings()
_REDIS_URL: str = _settings.REDIS_URL
_CACHE_PREFIX = "ecopackai:classify:"
_RATE_PREFIX = "ecopackai:rate:"
_DEFAULT_TTL = 3600  # seconds (1 hour)


# ---------------------------------------------------------------------------
# Internal Redis client (lazy, connection-pool backed)
# ---------------------------------------------------------------------------

_redis_pool: Optional[Any] = None


def _get_redis() -> Optional[Any]:
    """Return a shared async Redis client from a connection pool.

    Returns None if redis.asyncio is not installed or the URL is invalid.
    """
    global _redis_pool
    if not _HAS_REDIS:
        return None
    if _redis_pool is None:
        try:
            _redis_pool = aioredis.from_url(
                _REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        except Exception as exc:
            logger.warning("Could not create Redis pool: %s", exc)
    return _redis_pool


# ---------------------------------------------------------------------------
# Cache Key Builder
# ---------------------------------------------------------------------------

def _make_cache_key(product_features: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 cache key from a feature dict.

    Parameters
    ----------
    product_features : dict
        Product feature dictionary (order-independent via sort_keys).

    Returns
    -------
    str
        Full Redis key string: ``ecopackai:classify:<sha256>``.
    """
    serialised = json.dumps(product_features, sort_keys=True)
    digest = hashlib.sha256(serialised.encode()).hexdigest()
    return f"{_CACHE_PREFIX}{digest}"


def build_cache_key(product_features: Dict[str, Any]) -> str:
    """Public alias for _make_cache_key — usable by classify_api.py."""
    return _make_cache_key(product_features)


# ---------------------------------------------------------------------------
# Cache Operations
# ---------------------------------------------------------------------------

async def get_cached_classification(key: str) -> Optional[Dict[str, Any]]:
    """Retrieve a cached classification result.

    Parameters
    ----------
    key : str
        Redis key (from :func:`build_cache_key`).

    Returns
    -------
    dict or None
        Cached response dict on HIT, None on MISS or error.
    """
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = await r.get(key)
        if raw is None:
            return None
        result: Dict[str, Any] = json.loads(raw)
        logger.debug("Cache HIT: %s", key)
        return result
    except Exception as exc:
        logger.warning("Cache GET error: %s", exc)
        return None


async def set_cached_classification(
    key: str,
    value: Dict[str, Any],
    ttl_seconds: int = _DEFAULT_TTL,
) -> None:
    """Store a classification result in Redis.

    Parameters
    ----------
    key : str
        Redis key (from :func:`build_cache_key`).
    value : dict
        Serialisable classification result dict.
    ttl_seconds : int
        Time-to-live in seconds. Defaults to 3600 (1 hour).
    """
    r = _get_redis()
    if r is None:
        return
    try:
        await r.setex(key, ttl_seconds, json.dumps(value))
        logger.debug("Cache SET: %s (ttl=%ds)", key, ttl_seconds)
    except Exception as exc:
        logger.warning("Cache SET error: %s", exc)


async def invalidate_classification_cache() -> int:
    """Flush all classification cache entries (called on model promotion).

    Uses Redis SCAN to find all matching keys rather than a blocking KEYS
    command, making it safe for large key spaces.

    Returns
    -------
    int
        Number of keys deleted.
    """
    r = _get_redis()
    if r is None:
        return 0
    try:
        deleted = 0
        cursor = 0
        pattern = f"{_CACHE_PREFIX}*"
        while True:
            cursor, keys = await r.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                await r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info("Classification cache invalidated — %d keys removed.", deleted)
        return deleted
    except Exception as exc:
        logger.warning("Cache invalidation error: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

async def is_redis_connected() -> bool:
    """Ping Redis to verify connectivity.

    Returns
    -------
    bool
        True if Redis responds to PING within the socket timeout.
    """
    r = _get_redis()
    if r is None:
        return False
    try:
        await r.ping()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Rate Limiter (token bucket via Redis INCR + EXPIRE)
# ---------------------------------------------------------------------------

async def check_rate_limit(
    tenant_id: str,
    limit: int,
    window_seconds: int = 60,
) -> bool:
    """Enforce a token-bucket rate limit using Redis INCR + EXPIRE.

    On the first request within a window, sets an EXPIRE so the key
    auto-deletes after ``window_seconds``. Subsequent requests within that
    window increment the counter. Returns False when the limit is exceeded.

    Parameters
    ----------
    tenant_id : str
        Unique identifier for the rate-limit bucket (e.g. IP address or
        API key). Used to build the Redis key.
    limit : int
        Maximum number of requests allowed within ``window_seconds``.
    window_seconds : int
        Duration of the sliding window in seconds. Default: 60.

    Returns
    -------
    bool
        True if the request is within the rate limit, False if exceeded.
    """
    r = _get_redis()
    if r is None:
        # Redis unavailable — allow all requests (fail open)
        return True
    try:
        key = f"{_RATE_PREFIX}{tenant_id}"
        count = await r.incr(key)
        if count == 1:
            # First request in this window — set TTL
            await r.expire(key, window_seconds)
        return int(count) <= limit
    except Exception as exc:
        logger.warning("Rate limiter error: %s — allowing request.", exc)
        return True  # fail open
