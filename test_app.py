"""Unit tests for Kestrel claim-risk service input validation and behaviour."""
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}

def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Kestrel Claim Risk Review" in res.data

def test_valid_claim(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": 1328,
        "days_since_purchase": 120,
        "customer_prior_claims": 0,
        "partner_inspected": "N",
        "photo_attached": "Y"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "score" in data
    assert 0.0 <= data["score"] <= 1.0
    assert "flag_for_review" in data
    assert "reasons" in data
    assert len(data["reasons"]) > 0

def test_reject_nan_claim_amount_string(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": "nan",
        "days_since_purchase": 120,
        "customer_prior_claims": 0,
        "partner_inspected": "N",
        "photo_attached": "Y"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "claim_amount_inr must be a finite number" in res.get_json()["error"]

def test_reject_inf_claim_amount(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": "inf",
        "days_since_purchase": 120,
        "customer_prior_claims": 0,
        "partner_inspected": "N",
        "photo_attached": "Y"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "claim_amount_inr must be a finite number" in res.get_json()["error"]

def test_reject_negative_claim_amount(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": -500,
        "days_since_purchase": 120,
        "customer_prior_claims": 0,
        "partner_inspected": "N",
        "photo_attached": "Y"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "cannot be negative" in res.get_json()["error"]

def test_reject_negative_days(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": 1200,
        "days_since_purchase": -10,
        "customer_prior_claims": 0,
        "partner_inspected": "N",
        "photo_attached": "Y"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "days_since_purchase cannot be negative" in res.get_json()["error"]

def test_reject_negative_prior_claims(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": 1200,
        "days_since_purchase": 100,
        "customer_prior_claims": -1,
        "partner_inspected": "N",
        "photo_attached": "Y"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "customer_prior_claims cannot be negative" in res.get_json()["error"]

def test_reject_missing_required_field(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": 1200,
        "days_since_purchase": 100,
        # missing customer_prior_claims
        "partner_inspected": "N",
        "photo_attached": "Y"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "missing field(s)" in res.get_json()["error"]

def test_reject_invalid_flags(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": 1200,
        "days_since_purchase": 100,
        "customer_prior_claims": 0,
        "partner_inspected": "MAYBE",
        "photo_attached": "Y"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "partner_inspected must be 'Y' or 'N'" in res.get_json()["error"]

def test_reject_manipulated_submitted_at_date(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": 1200,
        "days_since_purchase": 100,
        "customer_prior_claims": 0,
        "partner_inspected": "N",
        "photo_attached": "Y",
        "submitted_at": "2024-01-01 10:00"  # Pre-regime historical date to manipulate regime
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "predates active policy regime" in res.get_json()["error"]

def test_reject_malformed_submitted_at_date(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": 1200,
        "days_since_purchase": 100,
        "customer_prior_claims": 0,
        "partner_inspected": "N",
        "photo_attached": "Y",
        "submitted_at": "not-a-real-date"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "invalid date format" in res.get_json()["error"]

def test_accept_valid_submitted_at_date(client):
    payload = {
        "partner_id": "DEMO_PARTNER_001",
        "claim_amount_inr": 1200,
        "days_since_purchase": 100,
        "customer_prior_claims": 0,
        "partner_inspected": "N",
        "photo_attached": "Y",
        "submitted_at": "2026-06-15 14:30"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "score" in res.get_json()

def test_reject_non_json(client):
    res = client.post("/predict", data="plain text data", content_type="text/plain")
    assert res.status_code == 400
