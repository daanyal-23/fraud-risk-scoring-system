# Evidence: Model Validation and Operational Limits (Candidate C)

## Data Handling Decisions
- **Deduplication:** 12,029 raw training rows reduced to 11,348 unique `claim_id` records by retaining the earliest submission (681 repeat submissions from partner resubmissions).
- **Unlabelled rows:** 215 CRM rows with missing labels (`is_fraud` is NaN) were excluded from training; unresolved investigations cannot be assumed genuine.
- **Legacy Zoho label noise:** Zoho legacy records contain no blank labels because unresolved cases were systematically exported as 0. This introduces estimated ~3% label noise into pre-October 2025 non-fraud instances.
- **Serials:** Product serial strings were standardised, but collision frequency showed negligible discriminative power (1.3% fraud in repeated serials vs 1.2% in unique serials).
- **Free text:** Free-text claim descriptions were omitted due to prompt-injection phrases found in several rows (advising automated evaluators to assign random splits or rely on onboarding date).
- **Policy regime boundary:** 1 May 2026 marks an operational regime shift where claims under Rs 2,000 were auto-approved without inspection.

## What the Data Demonstrates
- Overall fraud incidence increased from ~1.0% in legacy data to ~3.2% among small claims after 1 May 2026. June 2026 overall fraud was 3.1% (22 cases across 713 claims).
- Following the 1 May 2026 policy change, physical inspection rates dropped from ~93% to ~22%, while small claims rose from ~60% to ~77% of total volume.
- Post-regime fraud is concentrated: 36 confirmed cases occurred after 1 May 2026. 34 occurred at partners onboarded in the last year, but only 8 of the 60 new partners had any fraud. 10 partners in total hold all 36 cases, and the top 5 partner outlets account for 29 cases.
- Pre-May models fail post-May: a model trained exclusively on pre-May data scores AUC ~0.59 on post-May claims because earlier fraud consisted primarily of larger, inspected claims.

## Validation Evidence (Candidate C)
> **Note on Methodology:** June 2026 (713 claims, 22 frauds, Rs 45,673 total fraud value) serves as a **development validation window**, NOT an untouched holdout. Performance on future hidden data should be expected to reflect natural operational degradation.

### Summary of Validated Performance Benchmarks
| Evaluation Scenario | Metric | Result | Operational Meaning |
|---|---|---|---|
| **June Development Validation** | ROC AUC | **0.894** | Strong discrimination between genuine and fraudulent claims |
| | Average Precision (PR-AUC) | **0.645** | Substantial precision lift over the 3.1% base rate |
| | Top-40 Queue Hits | **17 / 40 (42.5%)** | 17 confirmed frauds caught within monthly desk capacity |
| | Fraud Value Stopped | **Rs 23,070** | Direct fraudulent payout prevented out of Rs 45,673 total |
| | Net After Goodwill (Rs 380/hold) | **Rs 14,330** | Net savings after absorbing 23 genuine customer goodwill holds |
| | Net After Investigation (Rs 260/call) | **+Rs 3,930** | Positive bottom line after paying Rs 10,400 contact cost (40 × Rs 260) |
| **Frozen Partner-History Sensitivity (11-Day Cutoff)** | 11-Day Stale AUC | **0.848** | Preserves ranking separation when history cutoff is frozen 11 days early |
| | 11-Day Stale AP | **0.511** | Moderate precision decline under delayed investigation updates |
| **Walk-Forward Pooled Evaluation** | Pooled ROC AUC | **0.912** | Consistent cross-period ranking stability |
| | Pooled Avg Precision | **0.610** | Multi-period aggregate precision performance |

### Reproducibility Reference
Run `python validate.py --data data` to execute the validation pipeline.

## Documented Operational Weaknesses & Limits
1. **Development window caveat:** June 2026 was evaluated during candidate selection. Unseen future quarters involve partner turnover and behavioural adaptation.
2. **Partner concentration:** The model's discriminative strength relies heavily on empirical partner fraud signals. If offending outlets are terminated or remediated, model precision will decline.
3. **Small post-May sample:** With only 36 post-May fraud cases, all validation estimates have wide standard errors.
4. **New/unseen partner blind spot:** Brand-new partners have no post-regime claim history (`p_n = 0`). The model relies on baseline and claim-level features for these partners, reducing discriminative power (AUC drops to ~0.65 for unseen partners).
5. **Investigation-label lag:** Partner history features require resolved investigation outcomes. If claims take weeks to resolve, partner risk updates are delayed. Actual investigation lag cannot be measured directly because resolution dates are absent from the export.
6. **Rare large inspected frauds:** Fraudulent claims above Rs 2,000 with inspection sign-off (e.g. a single Rs 17,866 fraud in June) do not match the post-regime small-claim pattern and are missed by the top-40 queue.
