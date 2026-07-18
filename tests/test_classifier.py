"""
test_classifier.py — Tests for the EcoPackAI Classification Service
====================================================================

Covers the FastAPI ``/v1/classify`` and ``/v1/health`` endpoints plus
input-validation edge cases.  Tests are designed to pass both when a
trained model is present and when it is absent (503 graceful degradation).

Author: EcoPackAI Team
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.classify_api import app, _load_model, _state

# ---------------------------------------------------------------------------
# Test client — ensure model is loaded before tests run
# ---------------------------------------------------------------------------
_load_model()  # Trigger model loading (normally done by lifespan)
client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model_is_loaded() -> bool:
    """Check if the model was successfully loaded."""
    return _state.get("model") is not None


# ═══════════════════════════════════════════════════════════════════════════
# 1. Valid input → correct output schema
# ═══════════════════════════════════════════════════════════════════════════

class TestValidInput:
    """POST /v1/classify with valid payloads."""

    def test_valid_input_returns_correct_schema(self) -> None:
        """A well-formed request should return FragilityResponse fields."""
        payload = {
            "length_cm": 15.0,
            "width_cm": 10.0,
            "height_cm": 8.0,
            "weight_g": 500.0,
            "material_type": "glass",
        }
        response = client.post("/v1/classify", json=payload)

        if _model_is_loaded():
            assert response.status_code == 200
            data = response.json()
            assert "tier" in data
            assert "confidence" in data
            assert "tier_label" in data
            assert "probabilities" in data
            assert 0 <= data["tier"] <= 3
            assert 0.0 <= data["confidence"] <= 1.0
            assert data["tier_label"] in ["None", "Low", "Medium", "Critical"]
            assert isinstance(data["probabilities"], dict)
            assert len(data["probabilities"]) == 4
        else:
            assert response.status_code == 503

    def test_all_material_types_accepted(self) -> None:
        """Every valid material_type should be accepted without 422."""
        materials = ["glass", "electronics", "apparel", "fragile_liquid", "standard"]
        for mat in materials:
            payload = {
                "length_cm": 20.0,
                "width_cm": 15.0,
                "height_cm": 10.0,
                "weight_g": 300.0,
                "material_type": mat,
            }
            response = client.post("/v1/classify", json=payload)
            assert response.status_code in (200, 503), (
                f"material_type='{mat}' returned {response.status_code}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Critical-tier items → high confidence
# ═══════════════════════════════════════════════════════════════════════════

class TestCriticalTier:
    """Products with fragile characteristics should be classified correctly."""

    def test_fragile_liquid_high_fragility(self) -> None:
        """Fragile-liquid items should score high on fragility tiers."""
        if not _model_is_loaded():
            pytest.skip("Model not loaded — cannot test prediction quality.")

        payload = {
            "length_cm": 8.0,
            "width_cm": 6.0,
            "height_cm": 5.0,
            "weight_g": 800.0,
            "material_type": "fragile_liquid",
        }
        response = client.post("/v1/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Fragile liquid items should be tier 2 or 3
        assert data["tier"] in [2, 3], (
            f"Expected tier 2 or 3 for fragile_liquid, got {data['tier']}"
        )
        assert data["confidence"] > 0.3

    def test_apparel_low_fragility(self) -> None:
        """Apparel items should typically be classified as low fragility."""
        if not _model_is_loaded():
            pytest.skip("Model not loaded — cannot test prediction quality.")

        payload = {
            "length_cm": 40.0,
            "width_cm": 30.0,
            "height_cm": 10.0,
            "weight_g": 200.0,
            "material_type": "apparel",
        }
        response = client.post("/v1/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["tier"] in [0, 1], (
            f"Expected tier 0 or 1 for apparel, got {data['tier']}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Invalid inputs → ValidationError (422)
# ═══════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    """Malformed requests should return 422 from Pydantic validation."""

    def test_negative_dimensions_raise_422(self) -> None:
        """Negative dimension values must be rejected."""
        payload = {
            "length_cm": -5.0,
            "width_cm": 10.0,
            "height_cm": 8.0,
            "weight_g": 500.0,
            "material_type": "glass",
        }
        response = client.post("/v1/classify", json=payload)
        assert response.status_code == 422

    def test_zero_weight_raises_422(self) -> None:
        """Zero weight must be rejected (gt=0 constraint)."""
        payload = {
            "length_cm": 15.0,
            "width_cm": 10.0,
            "height_cm": 8.0,
            "weight_g": 0.0,
            "material_type": "glass",
        }
        response = client.post("/v1/classify", json=payload)
        assert response.status_code == 422

    def test_invalid_material_type_raises_422(self) -> None:
        """Unknown material_type must be rejected."""
        payload = {
            "length_cm": 15.0,
            "width_cm": 10.0,
            "height_cm": 8.0,
            "weight_g": 500.0,
            "material_type": "plastic",
        }
        response = client.post("/v1/classify", json=payload)
        assert response.status_code == 422

    def test_missing_fields_raise_422(self) -> None:
        """Missing required fields must be rejected."""
        payload = {"length_cm": 15.0}
        response = client.post("/v1/classify", json=payload)
        assert response.status_code == 422

    def test_empty_body_raises_422(self) -> None:
        """Empty request body must be rejected."""
        response = client.post("/v1/classify", json={})
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# 4. Model not loaded → 503
# ═══════════════════════════════════════════════════════════════════════════

class TestModelUnavailable:
    """Service should degrade gracefully when model is absent."""

    def test_health_endpoint_always_responds(self) -> None:
        """GET /v1/health should always return 200."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data

    def test_health_reports_model_status(self) -> None:
        """Health response should accurately report model loading state."""
        response = client.get("/v1/health")
        data = response.json()
        if data["model_loaded"]:
            assert data["status"] == "healthy"
        else:
            assert data["status"] == "degraded"
