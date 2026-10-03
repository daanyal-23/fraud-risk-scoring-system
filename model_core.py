"""Shared feature + scoring logic (used by train.py and app.py)."""
import numpy as np, pandas as pd

REGIME_START = pd.Timestamp("2026-05-01")   # policy: claims < Rs2000 auto-approved w/o inspection
FEATURES = ["small_uninsp","d300","days_since_purchase","claim_amount_inr",
            "customer_prior_claims","photo","ptype","p_te","post","insp"]
PTYPES = {"authorised_service_centre":0,"franchise":1,"freelance_technician":2}
ALPHA = 2.0

def basic_features(df):
    df = df.copy()
    df["dt"] = pd.to_datetime(df["submitted_at"])
    df["small_uninsp"] = ((df.claim_amount_inr < 2000) & (df.partner_inspected != "Y")).astype(int)
    df["insp"] = (df.partner_inspected == "Y").astype(int)
    df["photo"] = (df.photo_attached == "Y").astype(int)
    df["d300"] = (df.days_since_purchase < 300).astype(int)
    df["post"] = (df.dt >= REGIME_START).astype(int)
    df["ptype"] = df.partner_type.map(PTYPES).fillna(0).astype(int)
    return df

def partner_stats(labelled):
    """Fraud counts per partner in the CURRENT regime only (pre-May history is a different world)."""
    post = labelled[labelled.dt >= REGIME_START]
    g = post.groupby("partner_id").is_fraud.agg(["sum","size"])
    return {"base": float(post.is_fraud.mean()),
            "partners": {k: [int(v["sum"]), int(v["size"])] for k, v in g.iterrows()}}

def add_partner_cols(df, stats):
    base = stats["base"]; P = stats["partners"]
    df = df.copy()
    df["p_fr"] = df.partner_id.map(lambda p: P.get(p, [0,0])[0]).astype(float)
    df["p_n"]  = df.partner_id.map(lambda p: P.get(p, [0,0])[1]).astype(float)
    df["p_te"] = (df.p_fr + ALPHA*base) / (df.p_n + ALPHA)
    return df

def expanding_partner_cols(train, pre_base=None):
    """Leak-free version for training rows: both partner and global history use only earlier claims."""
    tr = train.sort_values("dt").copy()
    post = (tr.dt >= REGIME_START)
    if pre_base is None:
        pre = (tr.dt < REGIME_START)
        pre_base = float(tr[pre].is_fraud.mean()) if pre.sum() > 0 else 0.0108
    fr = tr.is_fraud * post; n = post.astype(int)
    # Strictly expanding global base rate (avoids leaking future post-regime labels into early training rows)
    cum_fr_all = fr.cumsum() - fr
    cum_n_all = n.cumsum() - n
    expanding_base = (cum_fr_all + ALPHA * pre_base) / (cum_n_all + ALPHA)
    cs = fr.groupby(tr.partner_id).cumsum() - fr
    cn = n.groupby(tr.partner_id).cumsum() - n
    tr["p_fr"], tr["p_n"] = cs, cn
    tr["p_te"] = (cs + ALPHA * expanding_base) / (cn + ALPHA)
    return tr

def reasons(row, coefs, scaler_mean, scaler_scale, top=3):
    """Plain-English reasons from the logistic contributions."""
    x = np.array([row[f] for f in FEATURES], float)
    contrib = coefs * (x - scaler_mean) / scaler_scale
    text = {
      "small_uninsp": lambda r: "Small claim (< Rs 2,000) paid without inspection (policy since 1 May 2026)" if r["small_uninsp"] else "Claim was inspected or is above Rs 2,000",
      "p_te": lambda r: f"Partner {r['partner_id']} had {int(r['p_fr'])} confirmed fraud case(s) in {int(r['p_n'])} claims since May 2026" if r["p_n"] else f"Partner {r['partner_id']} has no confirmed history since May 2026",
      "p_fr": lambda r: f"Partner has {int(r['p_fr'])} confirmed fraud case(s) since May 2026",
      "p_n": lambda r: f"Partner has {int(r['p_n'])} labelled claims since May 2026",
      "d300": lambda r: "Claim made within 300 days of purchase (older claims were almost never fraud)" if r["d300"] else "Claim made 300+ days after purchase (rarely fraud)",
      "days_since_purchase": lambda r: f"{int(r['days_since_purchase'])} days since purchase",
      "claim_amount_inr": lambda r: f"Claim amount Rs {r['claim_amount_inr']:,.0f}",
      "customer_prior_claims": lambda r: f"Customer has {int(r['customer_prior_claims'])} earlier claim(s) (repeat claimants are riskier)",
      "photo": lambda r: "Photo attached" if r["photo"] else "No photo attached",
      "ptype": lambda r: "Partner type: " + {0:"authorised service centre",1:"franchise",2:"freelance technician"}[int(r["ptype"])],
      "post": lambda r: "Claim falls under the post-1-May-2026 approval rules" if r["post"] else "Claim predates the May 2026 rule change",
      "insp": lambda r: "Partner inspection sign-off present" if r["insp"] else "No partner inspection sign-off",
    }
    order = np.argsort(-np.abs(contrib))[:top]
    out = []
    for i in order:
        f = FEATURES[i]
        out.append({"factor": text[f](row), "direction": "raises risk" if contrib[i] > 0 else "lowers risk",
                    "weight": round(float(contrib[i]), 3)})
    return out
