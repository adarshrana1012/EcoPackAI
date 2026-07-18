"""
test_integration.py — Integration Tests for Phase 9 & 10 Endpoints (P15)
========================================================================

Tests cover:
  - Auth: POST /v1/auth/login with valid/invalid credentials
  - Protection: GET /v1/metrics/aggregate without token
  - Operations: GET /v1/shipments and GET /v1/shipments/export endpoints

Author: EcoPackAI Team
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient
from src.main import app

@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

@pytest.fixture
def auth_token(client):
    resp = client.post('/v1/auth/login', json={'email': 'demo@ecopackai.io', 'password': 'demo123'})
    assert resp.status_code == 200
    return resp.json()['access_token']

def test_auth_login_success(client):
    """POST /v1/auth/login returns a valid JWT token on correct credentials."""
    resp = client.post('/v1/auth/login', json={
        'email': 'demo@ecopackai.io',
        'password': 'demo123'
    })
    assert resp.status_code == 200
    data = resp.json()
    assert 'access_token' in data
    assert data['token_type'] == 'bearer'

def test_auth_login_invalid_credentials(client):
    """POST /v1/auth/login returns 401 Unauthorized on invalid passwords."""
    resp = client.post('/v1/auth/login', json={
        'email': 'demo@ecopackai.io',
        'password': 'wrong'
    })
    assert resp.status_code == 401
    detail = resp.json()['detail']
    assert 'Incorrect' in detail or 'Invalid' in detail

def test_health_probe(client):
    """GET /v1/health yields liveness state."""
    resp = client.get('/v1/health')
    assert resp.status_code == 200
    assert resp.json()['status'] == 'healthy'

def test_ready_probe(client):
    """GET /v1/ready yields readiness state."""
    resp = client.get('/v1/ready')
    # Can be 200 or 503 depending on if redis/db is running during tests
    assert resp.status_code in [200, 503]
    if resp.status_code == 200:
        assert resp.json()['ready'] is True

def test_startup_probe(client):
    """GET /v1/startup yields startup state."""
    resp = client.get('/v1/startup')
    assert resp.status_code in [200, 503]

def test_protected_endpoint_no_token(client):
    """Accessing protected routes without Bearer token returns 401."""
    resp = client.get('/v1/metrics/aggregate')
    assert resp.status_code == 401

def test_protected_endpoint_with_token(client, auth_token):
    """Accessing protected routes with Bearer token yields valid data."""
    headers = {'Authorization': f'Bearer {auth_token}'}
    resp = client.get('/v1/metrics/aggregate', headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert 'total_shipments' in data
    assert 'mean_void_pct' in data

def test_shipments_list_with_auth(client, auth_token):
    """GET /v1/shipments returns paginated list of shipments."""
    headers = {'Authorization': f'Bearer {auth_token}'}
    resp = client.get('/v1/shipments', headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert 'shipments' in data
    assert 'total' in data
    assert 'page' in data

def test_shipments_export_csv(client, auth_token):
    """GET /v1/shipments/export yields a CSV download file."""
    headers = {'Authorization': f'Bearer {auth_token}'}
    resp = client.get('/v1/shipments/export', headers=headers)
    assert resp.status_code == 200
    assert 'text/csv' in resp.headers['content-type']
    assert 'shipment_id' in resp.text
