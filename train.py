"""python train.py --data /path/to/pack   -> model.json, partner_stats.json, partner_lookup.json, predictions.csv"""
import argparse, json, os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from model_core import *

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default=".")
a = ap.parse_args()
D = lambda f: os.path.join(a.data, f)
part = pd.read_csv(D("partners.csv")); prod = pd.read_csv(D("products.csv"))
tr = pd.read_csv(D("train.csv")); te = pd.read_csv(D("test_unlabelled.csv"))
tr = tr.sort_values("submitted_at").drop_duplicates("claim_id", keep="first")   # partner re-submissions
tr = tr[tr.is_fraud.notna()]                                                    # undecided cases are not labels
tr = basic_features(tr.merge(part, on="partner_id", how="left"))
te = basic_features(te.merge(part, on="partner_id", how="left"))

# Candidate C: Post-1-May-2026 regime only
tr = tr[tr.dt >= REGIME_START]

stats = partner_stats(tr)
trx = expanding_partner_cols(tr)
sc = StandardScaler().fit(trx[FEATURES].astype(float))
lr = LogisticRegression(C=0.3, max_iter=1000).fit(sc.transform(trx[FEATURES].astype(float)), trx.is_fraud)

tex = add_partner_cols(te, stats)
tex["score"] = lr.predict_proba(sc.transform(tex[FEATURES].astype(float)))[:,1]
sub = pd.read_csv(D("sample_submission.csv"))[["claim_id"]].merge(tex[["claim_id","score"]], on="claim_id", how="left")
assert sub.score.notna().all() and len(sub) == len(te)
sub["score"] = sub.score.round(6)
sub.to_csv(os.path.join(a.out, "predictions.csv"), index=False)

json.dump({"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
           "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "features": FEATURES},
          open(os.path.join(a.out, "model.json"), "w"), indent=1)
json.dump(stats, open(os.path.join(a.out, "partner_stats.json"), "w"))
# partner lookup for the service (type only; no customer data)
part[["partner_id","city","partner_type"]].to_json(os.path.join(a.out, "partner_lookup.json"), orient="records")
print("trained on", len(trx), "rows,", int(trx.is_fraud.sum()), "fraud; wrote predictions for", len(sub))
print(sorted(zip(FEATURES, lr.coef_[0].round(2)), key=lambda x:-abs(x[1])))
