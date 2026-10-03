"""Candidate C / Acc 3 — Reproducible Validation & Evidence Script.
Usage:
    python validate.py --data ./data

Reproduces the Candidate C evaluation methodology:
  - Post-1-May-2026 regime only (1,422 claims, 36 fraud)
  - Train on May 2026 (709 claims, 14 fraud)
  - Development validation on June 2026 (713 claims, 22 fraud)
  - Frozen partner-history sensitivity test (0 to 14 days lag)
  - Walk-forward temporal pooled validation
All metrics are dynamically calculated from the data and model pipeline.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from model_core import (
    REGIME_START, FEATURES,
    basic_features, partner_stats, add_partner_cols, expanding_partner_cols
)

def validate_june_development(tr_post):
    """June 2026 development validation window (Candidate C methodology)."""
    train_may = tr_post[tr_post.dt < "2026-06-01"].copy()
    val_june = tr_post[tr_post.dt >= "2026-06-01"].copy()

    stats_may = partner_stats(train_may)
    trx_may = expanding_partner_cols(train_may)
    sc = StandardScaler().fit(trx_may[FEATURES].astype(float))
    lr = LogisticRegression(C=0.3, max_iter=1000).fit(sc.transform(trx_may[FEATURES].astype(float)), trx_may.is_fraud)

    june_x = add_partner_cols(val_june, stats_may)
    june_x["score"] = lr.predict_proba(sc.transform(june_x[FEATURES].astype(float)))[:, 1]

    auc = roc_auc_score(val_june.is_fraud, june_x.score)
    ap = average_precision_score(val_june.is_fraud, june_x.score)

    top40 = june_x.sort_values("score", ascending=False).head(40)
    hits = int(top40.is_fraud.sum())
    fraud_stopped = float(top40[top40.is_fraud == 1].claim_amount_inr.sum())
    genuine_holds = len(top40) - hits
    contact_cost = len(top40) * 260
    net_goodwill = fraud_stopped - genuine_holds * 380
    net_total = net_goodwill - contact_cost

    return {
        "train_rows": len(train_may),
        "train_fraud": int(train_may.is_fraud.sum()),
        "val_rows": len(val_june),
        "val_fraud": int(val_june.is_fraud.sum()),
        "auc": auc,
        "ap": ap,
        "hits": hits,
        "fraud_stopped": fraud_stopped,
        "genuine_holds": genuine_holds,
        "contact_cost": contact_cost,
        "net_goodwill": net_goodwill,
        "net_total": net_total,
        "scaler": sc,
        "model": lr
    }

def validate_lag_sensitivity(tr_post, sc, lr):
    """Evaluate performance when partner history is frozen or lagged."""
    val_june = tr_post[tr_post.dt >= "2026-06-01"].copy()
    results = []

    # Test lag cutoffs leading up to June
    for lag_days in [0, 3, 7, 11, 14]:
        cutoff = pd.Timestamp("2026-06-01") - pd.Timedelta(days=lag_days)
        stale_train = tr_post[tr_post.dt < cutoff]
        st = partner_stats(stale_train)
        jx = add_partner_cols(val_june, st)
        scores = lr.predict_proba(sc.transform(jx[FEATURES].astype(float)))[:, 1]
        auc = roc_auc_score(val_june.is_fraud, scores)
        ap = average_precision_score(val_june.is_fraud, scores)
        top40 = jx.assign(score=scores).sort_values("score", ascending=False).head(40)
        hits = int(top40.is_fraud.sum())
        stopped = float(top40[top40.is_fraud == 1].claim_amount_inr.sum())
        results.append({
            "lag_days": lag_days,
            "cutoff": cutoff.strftime("%Y-%m-%d"),
            "auc": auc,
            "ap": ap,
            "hits": hits,
            "stopped": stopped
        })
    return results

def validate_walk_forward(tr_post):
    """Walk-forward temporal evaluation across post-May sequential blocks."""
    splits = [
        ("2026-05-15", "2026-06-01"),
        ("2026-06-01", "2026-07-01")
    ]
    pooled_y, pooled_pred = [], []
    fold_results = []

    for t_end, v_end in splits:
        train_df = tr_post[tr_post.dt < t_end].copy()
        val_df = tr_post[(tr_post.dt >= t_end) & (tr_post.dt < v_end)].copy()

        st = partner_stats(train_df)
        tx = expanding_partner_cols(train_df)
        sc = StandardScaler().fit(tx[FEATURES].astype(float))
        lr = LogisticRegression(C=0.3, max_iter=1000).fit(sc.transform(tx[FEATURES].astype(float)), tx.is_fraud)

        vx = add_partner_cols(val_df, st)
        scores = lr.predict_proba(sc.transform(vx[FEATURES].astype(float)))[:, 1]

        auc_fold = roc_auc_score(val_df.is_fraud, scores)
        ap_fold = average_precision_score(val_df.is_fraud, scores)

        pooled_y.extend(val_df.is_fraud.tolist())
        pooled_pred.extend(scores.tolist())

        fold_results.append({
            "train_window": f"< {t_end}",
            "val_window": f"[{t_end} to {v_end})",
            "val_rows": len(val_df),
            "val_fraud": int(val_df.is_fraud.sum()),
            "auc": auc_fold,
            "ap": ap_fold
        })

    pooled_auc = roc_auc_score(pooled_y, pooled_pred)
    pooled_ap = average_precision_score(pooled_y, pooled_pred)

    return {
        "folds": fold_results,
        "pooled_auc": pooled_auc,
        "pooled_ap": pooled_ap
    }

def main():
    parser = argparse.ArgumentParser(description="Validate Candidate C Model")
    parser.add_argument("--data", default="data", help="Path to data directory")
    args = parser.parse_args()

    part = pd.read_csv(os.path.join(args.data, "partners.csv"))
    tr = pd.read_csv(os.path.join(args.data, "train.csv"))
    tr = tr.sort_values("submitted_at").drop_duplicates("claim_id", keep="first")
    tr = tr[tr.is_fraud.notna()]
    tr = basic_features(tr.merge(part, on="partner_id", how="left"))

    # Candidate C operates on the post-1-May-2026 regime ONLY
    tr_post = tr[tr.dt >= REGIME_START].copy()

    print("=" * 72)
    print("CANDIDATE C / ACC 3 — VALIDATION & EVIDENCE REPRODUCTION")
    print("=" * 72)
    print(f"Total post-May 2026 training claims: {len(tr_post)} (Frauds: {int(tr_post.is_fraud.sum())})")
    print("Regime start: 2026-05-01 (claims < Rs 2,000 auto-approved w/o inspection)\n")

    # 1. June Development Validation
    print("--- 1. JUNE DEVELOPMENT VALIDATION ---")
    june = validate_june_development(tr_post)
    print(f"Train (May 2026):     {june['train_rows']} claims, {june['train_fraud']} fraud")
    print(f"Val (June 2026):      {june['val_rows']} claims, {june['val_fraud']} fraud")
    print(f"ROC AUC:              {june['auc']:.4f} (rounds to {june['auc']:.3f})")
    print(f"Average Precision:    {june['ap']:.4f} (rounds to {june['ap']:.3f})")
    print(f"Top-40 Queue Hits:    {june['hits']} / 40 frauds (hit rate: {june['hits']/40*100:.1f}%)")
    print(f"Fraud Value Stopped:  Rs {june['fraud_stopped']:,.0f}")
    print(f"Genuine Holds:        {june['genuine_holds']} claims (goodwill cost: Rs {june['genuine_holds'] * 380:,.0f} @ Rs 380/hold)")
    print(f"Net After Goodwill:   Rs {june['net_goodwill']:,.0f}")
    print(f"Contact Cost:         Rs {june['contact_cost']:,.0f} ({june['hits'] + june['genuine_holds']} reviews @ Rs 260/contact)")
    print(f"Net Total Outcome:    Rs {june['net_total']:+,.0f}")

    # 2. Frozen Partner-History Sensitivity
    print("\n--- 2. FROZEN PARTNER-HISTORY SENSITIVITY ---")
    lag_res = validate_lag_sensitivity(tr_post, june["scaler"], june["model"])
    print(f"{'Cutoff Lead':<16}{'History Cutoff':<16}{'AUC':<9}{'Avg Prec':<11}{'Top-40':<9}{'Value Stopped'}")
    print("-" * 72)
    for r in lag_res:
        label = f"{r['lag_days']} days stale" if r['lag_days'] > 0 else "0 days (fresh)"
        print(f"{label:<16}{r['cutoff']:<16}{r['auc']:<9.3f}{r['ap']:<11.3f}{r['hits']:>2}/40    Rs {r['stopped']:,.0f}")
    print("Methodological note: Evaluates ranking sensitivity when partner history is frozen")
    print("0 to 14 days prior to June 1. Model weights are trained on May; does not perform")
    print("daily point-in-time model refits.")

    # 3. Walk-Forward Temporal Validation
    print("\n--- 3. WALK-FORWARD POOLED VALIDATION ---")
    wf = validate_walk_forward(tr_post)
    for i, fold in enumerate(wf["folds"], 1):
        print(f"Fold {i}: Train {fold['train_window']:<14} Val {fold['val_window']:<25} AUC={fold['auc']:.3f}, AP={fold['ap']:.3f}")
    print(f"\nWalk-Forward Pooled ROC AUC:  {wf['pooled_auc']:.4f} (rounds to {wf['pooled_auc']:.3f})")
    print(f"Walk-Forward Pooled Avg Prec: {wf['pooled_ap']:.4f} (rounds to {wf['pooled_ap']:.3f})")

    # 4. Methodological Notes & Known Limitations
    print("\n" + "=" * 72)
    print("METHODOLOGICAL CONTEXT & DOCUMENTED LIMITATIONS")
    print("=" * 72)
    print("1. Development window, not holdout: June is a development validation window")
    print("   used for candidate comparison, NOT an untouched holdout set.")
    print("2. Small fraud sample: Exactly 36 post-May fraud cases exist in total")
    print("   (14 in May, 22 in June), so confidence intervals are wide.")
    print("3. Partner concentration: Fraud is concentrated in a handful of partners")
    print("   (5 outlets hold 29 of the 36 post-May frauds).")
    print("4. New partner blind spot: Brand-new partners have no history and cannot")
    print("   be separated by partner features alone.")
    print("5. Investigation label lag: Model accuracy relies on timely closure of")
    print("   investigations; real-world lag is unknown because resolution dates are omitted.")
    print("6. Rare large inspected frauds: Fraudulent claims with inspection sign-off")
    print("   above Rs 2,000 are not caught by the small-claim review queue.")
    print("=" * 72)

if __name__ == "__main__":
    main()
