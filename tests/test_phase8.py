"""
test_phase8.py — Integration Tests for Phase 8 API Gateway (P07–P10)
====================================================================

Tests cover:
  - src/auth.py   : token creation, verification, user decode, admin guard
  - src/cache.py  : key building, cache round-trip, invalidation, rate limiter
  - src/database.py : mock fallback aggregate metrics, model building
  - src/main.py   : gateway smoke test (module imports)

Author: EcoPackAI Team
"""

from __future__ import annotations

import asyncio
import json
import pytest

# ---------------------------------------------------------------------------
# Auth Module Tests
# ---------------------------------------------------------------------------

class TestAuthModule:
    """Tests for src/auth.py."""

    def test_verify_password_correct(self):
        """verify_password returns True for matching passwords."""
        try:
            from src.auth import verify_password
        except ImportError:
            pytest.skip("passlib not installed")
        # verify_password with matching plain strings (fallback path)
        assert verify_password("test123", "test123") is True

    def test_verify_password_wrong(self):
        """verify_password returns False for mismatched passwords."""
        try:
            from src.auth import verify_password
        except ImportError:
            pytest.skip("passlib not installed")
        assert verify_password("wrong", "test123") is False

    def test_create_and_decode_token(self):
        """create_access_token + decode_access_token are inverses."""
        try:
            from src.auth import create_access_token, decode_access_token
        except ImportError:
            pytest.skip("python-jose not installed")
        token = create_access_token({"sub": "user@example.com", "role": "user"})
        payload = decode_access_token(token)
        assert payload["sub"] == "user@example.com"
        assert payload["role"] == "user"
        assert "exp" in payload
        assert "iat" in payload

    def test_login_response_model(self):
        """LoginResponse and LoginRequest Pydantic models are importable."""
        from src.auth import LoginRequest, LoginResponse
        req = LoginRequest(email="demo@ecopackai.io", password="demo123")
        assert req.email == "demo@ecopackai.io"
        resp = LoginResponse(access_token="tok123")
        assert resp.token_type == "bearer"

    def test_demo_users_exist(self):
        """Demo user store contains the expected accounts."""
        from src.auth import _DEMO_USERS
        assert "demo@ecopackai.io" in _DEMO_USERS
        assert "admin@ecopackai.io" in _DEMO_USERS
        assert _DEMO_USERS["admin@ecopackai.io"]["role"] == "admin"
        assert _DEMO_USERS["demo@ecopackai.io"]["role"] == "user"
        # Passwords stored as plain text for demo (no import-time hashing)
        assert _DEMO_USERS["demo@ecopackai.io"]["plain_password"] == "demo123"
        assert _DEMO_USERS["admin@ecopackai.io"]["plain_password"] == "admin123"


# ---------------------------------------------------------------------------
# Cache Module Tests
# ---------------------------------------------------------------------------

class TestCacheModule:
    """Tests for src/cache.py."""

    def test_build_cache_key_deterministic(self):
        """The same features always produce the same cache key."""
        from src.cache import build_cache_key
        features = {"length_cm": 15.0, "width_cm": 10.0, "height_cm": 8.0,
                    "weight_g": 500.0, "material_type": "glass"}
        key1 = build_cache_key(features)
        key2 = build_cache_key(features)
        assert key1 == key2

    def test_build_cache_key_order_independent(self):
        """Key is identical regardless of feature dict insertion order."""
        from src.cache import build_cache_key
        features_a = {"a": 1, "b": 2, "c": 3}
        features_b = {"c": 3, "a": 1, "b": 2}
        assert build_cache_key(features_a) == build_cache_key(features_b)

    def test_build_cache_key_prefix(self):
        """Cache key contains the expected namespace prefix."""
        from src.cache import build_cache_key, _CACHE_PREFIX
        key = build_cache_key({"x": 1})
        assert key.startswith(_CACHE_PREFIX)

    def test_build_cache_key_sha256_length(self):
        """SHA-256 hex digest is 64 characters."""
        from src.cache import build_cache_key, _CACHE_PREFIX
        key = build_cache_key({"x": 1})
        sha_part = key[len(_CACHE_PREFIX):]
        assert len(sha_part) == 64

    def test_different_features_different_keys(self):
        """Different feature vectors produce different cache keys."""
        from src.cache import build_cache_key
        key_a = build_cache_key({"material_type": "glass"})
        key_b = build_cache_key({"material_type": "apparel"})
        assert key_a != key_b

    def test_cache_miss_without_redis(self):
        """get_cached_classification returns None when Redis is unavailable."""
        import asyncio
        from src.cache import get_cached_classification
        result = asyncio.run(get_cached_classification("ecopackai:classify:nonexistent"))
        assert result is None

    def test_is_redis_connected_false_without_redis(self):
        """is_redis_connected returns a bool when Redis is not running."""
        import asyncio
        from src.cache import is_redis_connected
        result = asyncio.run(is_redis_connected())
        assert isinstance(result, bool)  # doesn't raise

    def test_rate_limit_passes_without_redis(self):
        """check_rate_limit fails open (returns True) when Redis is unavailable."""
        import asyncio
        from src.cache import check_rate_limit
        result = asyncio.run(check_rate_limit("test-tenant", limit=10, window_seconds=60))
        assert result is True  # fail open


# ---------------------------------------------------------------------------
# Database Module Tests
# ---------------------------------------------------------------------------

class TestDatabaseModule:
    """Tests for src/database.py."""

    def test_get_aggregate_metrics_fallback(self):
        """get_aggregate_metrics returns mock data when DB is unavailable."""
        import asyncio
        from src.database import get_aggregate_metrics
        result = asyncio.run(get_aggregate_metrics(db=None))
        assert result.total_shipments > 0
        assert result.mean_void_pct > 0
        assert len(result.weekly_material_kg) > 0
        assert "None" in result.fragility_distribution or "Low" in result.fragility_distribution

    def test_get_shipment_returns_none_without_db(self):
        """get_shipment returns None when DB is unavailable."""
        import asyncio
        from src.database import get_shipment
        result = asyncio.run(get_shipment(db=None, shipment_id="00000000-0000-0000-0000-000000000001"))
        assert result is None

    def test_create_shipment_no_op_without_db(self):
        """create_shipment silently skips when DB is unavailable."""
        import asyncio
        from src.database import create_shipment
        result = asyncio.run(create_shipment(
            db=None,
            order_id="order-test",
            box_sku="SM-01",
            void_pct=20.5,
            material_weight_g=150.0,
            co2e_kg=0.05,
        ))
        assert result is None

    def test_orm_models_importable(self):
        """ORM model classes are importable."""
        try:
            from src.database import ProductRecord, ShipmentRecord, PackingPolicyRecord
        except ImportError:
            pytest.skip("sqlalchemy[asyncio] not installed")
        assert ProductRecord.__tablename__ == "products"
        assert ShipmentRecord.__tablename__ == "shipments"
        assert PackingPolicyRecord.__tablename__ == "packing_policies"

    def test_pydantic_response_models(self):
        """Pydantic response models accept expected fields."""
        from src.database import ShipmentMetrics, AggregateMetrics
        sm = ShipmentMetrics(
            shipment_id="abc123",
            order_id="order-1",
            box_sku="LG-02",
            void_volume_pct=22.5,
            material_weight_g=200.0,
            co2e_kg=0.08,
            damage_reported=False,
            packed_at="2026-06-01T12:00:00Z",
        )
        assert sm.shipment_id == "abc123"

        am = AggregateMetrics(
            total_shipments=100,
            mean_void_pct=18.5,
            total_material_weight_kg=50.0,
            total_co2e_kg=100.0,
            weekly_material_kg=[{"week": "W1", "kg": 10.0}],
            fragility_distribution={"None": 80, "Low": 20},
        )
        assert am.total_shipments == 100


# ---------------------------------------------------------------------------
# Gateway Import Smoke Tests
# ---------------------------------------------------------------------------

class TestGatewayImport:
    """Verify that main.py imports cleanly without a live DB/Redis."""

    def test_auth_imports(self):
        """Auth module imports without errors."""
        from src.auth import (
            login_handler, get_current_user, require_admin,
            LoginRequest, LoginResponse, oauth2_scheme,
        )
        assert callable(login_handler)
        assert callable(get_current_user)
        assert callable(require_admin)

    def test_cache_imports(self):
        """Cache module imports without errors."""
        from src.cache import (
            get_cached_classification, set_cached_classification,
            invalidate_classification_cache, check_rate_limit,
            is_redis_connected, build_cache_key,
        )
        assert callable(get_cached_classification)
        assert callable(build_cache_key)

    def test_database_imports(self):
        """Database module imports without errors."""
        from src.database import (
            get_db, get_shipment, get_aggregate_metrics,
            create_shipment, update_shipment_damage,
        )
        assert callable(get_db)
        assert callable(create_shipment)

    def test_settings_has_jwt_fields(self):
        """Settings class exposes the new JWT and CORS fields."""
        from src.settings import Settings
        s = Settings()
        assert hasattr(s, "JWT_SECRET_KEY")
        assert hasattr(s, "JWT_ALGORITHM")
        assert hasattr(s, "JWT_EXPIRE_MINUTES")
        assert hasattr(s, "CORS_ORIGINS")
        assert s.JWT_ALGORITHM == "HS256"
        assert s.JWT_EXPIRE_MINUTES == 60
