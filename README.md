# Kestrel Claim-Risk Prioritisation Service (Candidate C)

Prioritises incoming warranty claims for Kestrel's 40-claims-per-month investigation desk. Operates entirely locally with zero external API calls, zero paid services, and zero network dependencies.

---

## 1. Data Privacy & Architecture Overview

> **Confidentiality Notice:**
> - Raw client data (`train.csv`, `test_unlabelled.csv`, `partners.csv`, `products.csv`, etc.) is strictly private and excluded from this public repository.
> - Model artifacts (`model.json`, `partner_stats.json`, `partner_lookup.json`) and predictions (`predictions.csv`) are derived from private client data and are also gitignored to protect client confidentiality.
> - Consequently, running or retraining the service on a new clone requires placing the private data pack into `./data` and running `train.py` to generate the necessary local artifacts.

---

## 2. Setup & Execution Workflow

### Step 1: Environment Setup (Python 3.10+)
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```
*(Serving requires `flask`, `numpy`, and `pandas`; training and testing also utilize `scikit-learn` and `pytest`.)*

### Step 2: Generate Model Artifacts (Requires Private Data Pack)
Place the private data files into `./data/`, then execute:
```bash
# Trains Candidate C (post-1-May-2026 regime) and writes:
# - model.json
# - partner_stats.json
# - partner_lookup.json
# - predictions.csv
python train.py --data ./data --out .
```

### Step 3: Run the Scoring Service
Once the model artifacts are generated:
```bash
python app.py
```
Access the review UI at `http://127.0.0.1:5000`.

### Scoring Endpoint (`POST /predict`)
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "partner_id": "DEMO_PARTNER_001",
    "claim_amount_inr": 1500,
    "days_since_purchase": 120,
    "customer_prior_claims": 0,
    "partner_inspected": "N",
    "photo_attached": "Y"
  }'
```

**Response Format:**
```json
{
  "score": 0.0452,
  "flag_for_review": false,
  "threshold": 0.30,
  "reasons": [
    {
      "factor": "Small claim (< Rs 2,000) paid without inspection (policy since 1 May 2026)",
      "direction": "raises risk",
      "weight": 0.412
    }
  ],
  "notes": ["Partner ID not in the partner list - treated as having no history."],
  "partner_city": null,
  "disclaimer": "Risk score for prioritising the 40-a-month review queue. Not proof of fraud; do not deny a claim on this alone."
}
```

**Input Validation & Error Handling:**
- Invalid non-finite numeric values (`NaN`, `inf`, `-inf`) return HTTP 422.
- Negative amounts, days, or prior claims return HTTP 422.
- Missing required fields return HTTP 422.
- Non-JSON or malformed request payloads return HTTP 400.
- Dates predating the 1 May 2026 policy change are rejected with HTTP 422 to prevent rule evasion.
- The service validates input cleanly and returns descriptive HTTP 400/422 responses on invalid data rather than crashing.

---

## 3. Validation & Evidence Reproduction

To reproduce the Candidate C validation metrics dynamically from data:

```bash
python validate.py --data ./data
```

**Candidate C Dynamically Calculated Results:**
- **June Development Validation (May train -> June val):**
  - ROC AUC: **0.894** (0.8936)
  - Average Precision: **0.645** (0.6449)
  - Top-40 Queue Fraud Hits: **17 / 40** (42.5% hit rate)
  - Direct Fraud Value Stopped: **Rs 23,070**
  - Net After Customer Goodwill (23 genuine holds @ Rs 380/hold): **Rs 14,330**
  - Net Total After Investigation Contact Cost (40 reviews @ Rs 260/contact): **+Rs 3,930**
- **Frozen Partner-History Sensitivity (11-Day Cutoff at 2026-05-21):**
  - ROC AUC: **0.848**
  - Average Precision: **0.511**
- **Walk-Forward Pooled Evaluation:**
  - Pooled ROC AUC: **0.912** (0.9118)
  - Pooled Average Precision: **0.610** (0.6101)

*(Note: June 2026 is an iterative development validation window, NOT an untouched holdout set.)*

---

## 4. Automated Tests

Run unit tests covering input validation, error responses, edge cases, and endpoints:
```bash
pytest -v test_app.py
```

---

## 5. Repository Structure & Artifact Privacy

| File / Directory | Description | Public / Private |
|---|---|---|
| `app.py` | Flask scoring service with robust validation | Public code |
| `model_core.py` | Shared feature engineering and explanation logic | Public code |
| `train.py` | Training pipeline for Candidate C model (post-May regime) | Public code |
| `validate.py` | Validation and evidence reproduction script | Public code |
| `test_app.py` | Test suite for input validation and API behavior | Public code |
| `static/index.html` | Front-end claim review UI | Public code |
| `requirements.txt` | Package dependencies | Public code |
| `MEMO.md` | Executive memo for Head of D2C Operations | Confidential review |
| `EVIDENCE.md` | Detailed analytical validation and operational limits | Confidential review |
| `submission-form.md` | Submission questions and responses | Confidential review |
| `data/` | Raw client CSVs, policy PDF, email threads | **PRIVATE (gitignored)** |
| `model.json` | Trained logistic regression weights and scaling parameters | **PRIVATE (gitignored)** |
| `partner_stats.json` | Partner fraud rates and claim counts | **PRIVATE (gitignored)** |
| `partner_lookup.json` | Partner type and metadata lookup | **PRIVATE (gitignored)** |
| `predictions.csv` | Scored predictions for unlabelled test claims | **PRIVATE (gitignored)** |

---

## 6. Three Critical Operational Rules

1. **Partner Watchlist Nature:** Since the 1 May 2026 rule change, post-regime fraud is concentrated in a handful of outlets. The model must be **retrained monthly** (`train.py`) as investigation outcomes arrive to avoid performance decay.
2. **New Partner Blind Spot:** The model relies on post-May partner track record. Exactly **11 partners (accounting for 49 test claims)** joined after 30 June 2026; brand-new partners have no history and score as "unknown", not "clean". Standard operational diligence must be applied to new outlets.
3. **Metric Focus:** Do not track overall classification accuracy (approving every claim already achieves ~97% accuracy while stopping zero fraud). Track **precision in the top-40 monthly review queue** and **net rupees stopped**.
