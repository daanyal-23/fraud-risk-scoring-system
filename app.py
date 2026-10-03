"""Kestrel claim-risk service.  python app.py  ->  http://127.0.0.1:5000
No API key, no network calls, no paid services. Only needs: flask, numpy, pandas."""
import json, math, os, numpy as np, pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from model_core import *

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f)))
M, STATS, LOOKUP = J("model.json"), J("partner_stats.json"), {r["partner_id"]: r for r in J("partner_lookup.json")}
COEF, MEAN, SCALE = np.array(M["coef"]), np.array(M["mean"]), np.array(M["scale"])
THRESHOLD = 0.30   # roughly the top ~3% of claims; see EVIDENCE.md for why
REQUIRED = ["partner_id", "claim_amount_inr", "days_since_purchase", "partner_inspected", "photo_attached", "customer_prior_claims"]

app = Flask(__name__, static_folder=os.path.join(HERE, "static"))

def score_record(rec):
    if not isinstance(rec, dict):
        raise ValueError("Claim record must be a JSON object / dict")

    # Validate required fields presence
    missing = [k for k in REQUIRED if rec.get(k) is None or (isinstance(rec.get(k), str) and rec.get(k).strip() == "")]
    if missing:
        raise ValueError("missing field(s): " + ", ".join(missing))

    # Validate partner_id
    partner_id = rec.get("partner_id")
    if not isinstance(partner_id, str) or not partner_id.strip():
        raise ValueError("partner_id must be a non-empty string")
    partner_id = partner_id.strip()

    # Validate claim_amount_inr
    raw_amount = rec.get("claim_amount_inr")
    if isinstance(raw_amount, bool):
        raise ValueError("claim_amount_inr must be a numeric value, not boolean")
    try:
        amount = float(raw_amount)
    except (ValueError, TypeError):
        raise ValueError(f"claim_amount_inr must be a valid number, got: {raw_amount}")
    if not math.isfinite(amount):
        raise ValueError("claim_amount_inr must be a finite number (cannot be NaN or Infinity)")
    if amount < 0:
        raise ValueError(f"claim_amount_inr cannot be negative, got: {amount}")

    # Validate days_since_purchase
    raw_days = rec.get("days_since_purchase")
    if isinstance(raw_days, bool):
        raise ValueError("days_since_purchase must be an integer, not boolean")
    try:
        days = float(raw_days)
    except (ValueError, TypeError):
        raise ValueError(f"days_since_purchase must be a valid integer, got: {raw_days}")
    if not math.isfinite(days):
        raise ValueError("days_since_purchase must be a finite number (cannot be NaN or Infinity)")
    if days < 0:
        raise ValueError(f"days_since_purchase cannot be negative, got: {days}")
    if not days.is_integer():
        raise ValueError(f"days_since_purchase must be a whole integer, got: {raw_days}")
    days = int(days)

    # Validate customer_prior_claims
    raw_prior = rec.get("customer_prior_claims")
    if isinstance(raw_prior, bool):
        raise ValueError("customer_prior_claims must be an integer, not boolean")
    try:
        prior = float(raw_prior)
    except (ValueError, TypeError):
        raise ValueError(f"customer_prior_claims must be a valid integer, got: {raw_prior}")
    if not math.isfinite(prior):
        raise ValueError("customer_prior_claims must be a finite number (cannot be NaN or Infinity)")
    if prior < 0:
        raise ValueError(f"customer_prior_claims cannot be negative, got: {prior}")
    if not prior.is_integer():
        raise ValueError(f"customer_prior_claims must be a whole integer, got: {raw_prior}")
    prior = int(prior)

    # Validate partner_inspected & photo_attached
    flags = {}
    for k in ("partner_inspected", "photo_attached"):
        val = rec.get(k)
        if not isinstance(val, str) or val.strip().upper() not in ("Y", "N"):
            raise ValueError(f"{k} must be 'Y' or 'N', got: {val}")
        flags[k] = val.strip().upper()

    # Validate submitted_at safely:
    # If provided, validate datetime parsing and disallow manipulating post-regime policy
    raw_submitted = rec.get("submitted_at")
    if raw_submitted is not None and str(raw_submitted).strip() != "":
        try:
            sub_dt = pd.to_datetime(raw_submitted)
            if pd.isna(sub_dt):
                raise ValueError("submitted_at is not a valid datetime")
        except Exception:
            raise ValueError(f"submitted_at has an invalid date format: {raw_submitted}")
        if sub_dt < REGIME_START:
            raise ValueError(f"submitted_at ({raw_submitted}) predates active policy regime ({REGIME_START.strftime('%Y-%m-%d')}); historical dates cannot alter current regime logic")
        submitted_at_str = sub_dt.strftime("%Y-%m-%d %H:%M")
    else:
        submitted_at_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    r = {
        "partner_id": partner_id,
        "claim_amount_inr": amount,
        "days_since_purchase": days,
        "customer_prior_claims": prior,
        "partner_inspected": flags["partner_inspected"],
        "photo_attached": flags["photo_attached"],
        "submitted_at": submitted_at_str
    }

    p = LOOKUP.get(partner_id)
    r["partner_type"] = p["partner_type"] if p else "authorised_service_centre"

    df = add_partner_cols(basic_features(pd.DataFrame([r])), STATS)
    row = df.iloc[0]
    x = np.array([row[f] for f in FEATURES], float)
    z = M["intercept"] + float(COEF @ ((x - MEAN) / SCALE))
    score = 1.0 / (1.0 + np.exp(-z))

    notes = []
    if not p:
        notes.append("Partner ID not in the partner list - treated as having no history.")
    if row.p_n == 0:
        notes.append("No labelled history for this partner since May 2026: the model is weakest on brand-new partners.")

    return {
        "score": round(float(score), 4),
        "flag_for_review": bool(score >= THRESHOLD),
        "threshold": THRESHOLD,
        "reasons": reasons(row, COEF, MEAN, SCALE),
        "notes": notes,
        "partner_city": p["city"] if p else None,
        "disclaimer": "Risk score for prioritising the 40-a-month review queue. Not proof of fraud; do not deny a claim on this alone."
    }

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    rec = request.get_json(silent=True)
    if not isinstance(rec, dict):
        return jsonify(error="send one claim as a JSON object"), 400
    try:
        return jsonify(score_record(rec))
    except (ValueError, TypeError) as e:
        return jsonify(error=str(e)), 422
    except Exception as e:
        return jsonify(error="internal server error"), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
