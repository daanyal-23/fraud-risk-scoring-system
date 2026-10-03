**To:** Ritu Deshpande, Head of D2C Operations  
**Cc:** Farhan Sheikh, Meenal Joshi, Tanmay Kulkarni  
**Re:** Warranty fraud: what the data says and what to do next week (Candidate C Model)

**The decision.** Don't rely on an automated claim-by-claim scoring model as our sole defence. Address the process leak first: since the 1 May change, claims under Rs 2,000 have been auto-approved without inspection, and confirmed post-May fraud is heavily concentrated in a small group of service outlets. Use Candidate C strictly as a prioritisation tool for the investigation desk's 40-claims-per-month capacity.

**Why not accuracy.** In June, only ~3.1% of claims were confirmed fraud (about 1 in 32). A naive rule that approves 100% of claims already achieves 96.9% accuracy while catching zero fraud. The board's 97% accuracy KPI would be satisfied by doing nothing. We recommend reporting two actionable metrics instead: how many confirmed frauds are identified among the 40 reviewed claims (precision in top 40), and net rupees of fraud stopped.

**What the data says about partners.** Concentrated, not category-wide. Since 1 May, 34 of 36 post-regime frauds occurred at partners onboarded over the last year. However, only 8 of the 60 newly onboarded partners show any fraud at all; the remaining 52 are clean. Furthermore, 5 specific outlets account for 29 of the 36 cases. The issue is bad actors at specific outlets, not all new partners. Outlets with very few claims need manual verification before punitive action.

**The numbers (Candidate C).** We validated the model using June 2026 claims as our development validation window (713 claims, 22 confirmed frauds totalling Rs 45,673). This is a development validation window, not an untouched future holdout.
- **Top 40 review performance:** Prioritising the top 40 claims stopped **17 frauds** (42.5% precision) and identified **Rs 23,070** in fraudulent claim value.
- **Economics:** Holding the 23 genuine claims incurred Rs 8,740 in customer goodwill cost (at Rs 380 each), leaving **Rs 14,330 net**. Factoring in investigation contact costs (40 reviews at Rs 260 each = Rs 10,400), the net financial outcome is **+Rs 3,930**.
- **Lag / stale history resilience:** When simulated with a frozen partner history (history cutoff set to 2026-05-21, 11 days prior to June 1), Candidate C maintained an AUC of 0.848 (average precision 0.511), catching 14 frauds in the top 40 (Rs 20,290 stopped) and demonstrating operational stability between monthly retraining cycles.

**Key operational limitations to acknowledge honestly:**
1. *Development window, not clean holdout:* June data was used iteratively during model development; true performance on future unlabelled quarters will face natural drift.
2. *New-partner weakness:* Candidate C relies significantly on post-regime partner history. Brand-new partners with zero historical claims cannot be flagged by partner track record alone.
3. *Investigation lag:* The model requires closed investigation outcomes to update partner profiles. Because resolution timestamps are not captured in the export, real-world lag remains an unmeasured operational variable.
4. *Rare large frauds missed:* Fraudulent claims above Rs 2,000 with inspection sign-off (such as a single Rs 17,866 fraud in June) do not exhibit the small-uninspected pattern and are not caught by this queue.
5. *Small sample noise:* With only 36 post-May fraud cases in total, all validation metrics carry wide confidence intervals.

**Recommendations for next week:**
1. **Reinstate targeted inspection:** Reinstate mandatory physical inspection for claims under Rs 2,000 specifically at high-risk flagged outlets. While reinstating inspection involves operational coordination and scheduling overhead, targeting only flagged outlets keeps costs contained.
2. **Review flagged outlets manually:** Have the service desk manually review pending claims from outlets showing initial fraud signals before taking partner-level contract steps.
3. **Data cleanup with IT:** Work with Tanmay to separate undecided claims from clean claims in CRM/legacy feeds, and clean anomalous text in claim descriptions.
4. **Re-evaluate the blanket Rs 2,000 threshold:** The blanket auto-approval rule increased small-claim fraud from 0.5% to >3%. Consider lowering the auto-approval threshold or adding basic partner tenure criteria.
5. **Monthly retrain:** Retrain Candidate C monthly as new investigation closures arrive to prevent partner features from going stale.

**Cost.** The model runs locally via standard open-source libraries: Rs 0 in third-party API or per-prediction costs. Hosting requires only a lightweight internal server.
