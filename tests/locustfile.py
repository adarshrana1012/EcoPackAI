"""
load_test.py — Locust load test for EcoPackAI API
"""

import random
from locust import HttpUser, task, between

MATERIALS = ['glass', 'electronics', 'apparel', 'standard', 'fragile_liquid']

def random_product():
    return {
        'length_cm': round(random.uniform(5, 50), 1),
        'width_cm': round(random.uniform(5, 50), 1),
        'height_cm': round(random.uniform(5, 50), 1),
        'weight_g': round(random.uniform(100, 5000), 1),
        'material_type': random.choice(MATERIALS),
    }

def random_order(n_items=None):
    n = n_items or random.randint(1, 3)
    return {'items': [random_product() for _ in range(n)]}


class EcoPackUser(HttpUser):
    wait_time = between(0.5, 2.0)
    token = None
    headers = {}

    def on_start(self):
        resp = self.client.post('/v1/auth/login', json={
            'email': 'demo@ecopackai.io',
            'password': 'demo123'
        })
        if resp.status_code == 200:
            self.token = resp.json()['access_token']
            self.headers = {'Authorization': f'Bearer {self.token}'}

    @task(5)
    def classify_product(self):
        # Target: P95 < 200ms
        self.client.post('/v1/classify', json=random_product(), headers=self.headers, name='/v1/classify')

    @task(3)
    def pack_order(self):
        # Target: P95 < 500ms
        self.client.post('/v1/pack', json=random_order(), headers=self.headers, name='/v1/pack')

    @task(2)
    def health_check(self):
        self.client.get('/v1/gateway/health', name='/v1/gateway/health')

    @task(1)
    def get_analytics(self):
        self.client.get('/v1/metrics/aggregate', headers=self.headers, name='/v1/metrics/aggregate')
