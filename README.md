# RiskGuard — AI Risk Manager for Return Fraud & Abuse Rings

**Track 02: AI Risk Manager** — a working detector for one class of loss (return fraud),
plus a second independent detector for coordinated abuse rings, with measured
precision/recall, honest false-positive cost accounting, and a defense-only scope.

Built for a buildathon deadline, and built to survive scrutiny: every number in this
repo is traceable to a script that produced it, every assumption is a comment where
it's made, and every bug we found along the way is documented rather than smoothed
over. That last part is deliberate — catching your own mistakes is a stronger signal
of ML maturity than a suspiciously clean first-try result.

---

## What this actually is

Two independently-evaluated detectors sharing one philosophy: **interpretable,
cost-aware, capacity-aware.**

1. **Return-Risk Scorer** — given a return is happening, how likely is it abusive
   (wardrobing, bracketing, serial returning)? XGBoost, isotonic-calibrated, with a
   cost-sensitive, capacity-constrained decision threshold — not a naive 0.5 cutoff.
2. **Abuse-Ring Sentinel** — does a cluster of accounts sharing a device/address/
   payment fingerprint look like one operator running several "distinct" accounts,
   or just benign sharing (a family at one address)? Graph clustering + a logistic
   classifier, evaluated separately from the return-risk model.

Both ship with plain-English explanations (deterministic, template-based — not an
LLM call, because a risk decision needs a reproducible, auditable explanation, not
one that could paraphrase differently on retry or hallucinate a reason the model
didn't actually use).

## Why synthetic data, stated up front

No real merchant or transaction data was available to us. Rather than use the
Kaggle credit-card-fraud dataset (heavily overused in hackathons, unrealistically
clean, PCA-transformed features), we built a **documented generative model**:
customer behavior types (normal / occasional returner / serial abuser / ring
member) drive realistic order and return patterns, with injected noise so a
perfect classifier is impossible. Every assumption lives as a comment in
`data/generate_data.py`, not a black box.

This means our metrics describe how well the models recover *this generative
rule* — not a guarantee about real-world fraud rates. We say so again wherever
it matters, rather than letting a strong number imply more than it should.

---

## Headline results (all on held-out test data)

**Return-Risk Scorer**
| Metric | Value |
|---|---|
| ROC-AUC (calibrated XGBoost) | 0.885 |
| Average Precision | 0.705 |
| Deployed threshold | 0.13 (capacity-constrained: max 20% of returns flaggable) |
| Precision / Recall at deployed threshold | 0.585 / 0.731 (95% CI: 0.536–0.632 / 0.681–0.780) |
| Savings vs. a rule-based baseline | ₹340,508 on a 90-day test window (12,489 orders) |
| Extrapolated savings per order | ₹27.26 |
| Projected annual savings, 50K orders/month merchant | ~₹1.6 crore *(linear extrapolation — see caveats below)* |

**Abuse-Ring Sentinel**
| Metric | Value |
|---|---|
| ROC-AUC / Avg Precision | 1.000 / 1.000 *(small sample — see caveats)* |
| Repeated-split recall (the honest version) | 0.997 (95% CI: 0.929–1.000) |
| Deployed threshold | 0.27 (best-F1 on held-out set) |

The perfect-looking ring-detector AUC is explained, not hidden, in
[notebook 04](notebooks/04_abuse_ring_detector.ipynb) — it's a real artifact of a
~50-cluster test set and a generator-imposed size gap between rings (3–9 members)
and benign groups (2–3 members), not a claim that real-world rings separate this
cleanly.

---

## What broke, and how we caught it

We're including this section on purpose. A track that explicitly rewards "honest
metrics" should get an honest account of the process, not just the polished output.

1. **The abuse-ring data was trivially separable by construction.** The first
   version of the generator made benign sharing (address only) and real rings
   (device+payment only) mutually exclusive sets — a classifier could hit perfect
   precision/recall just by checking one boolean. Caught before training, fixed by
   deliberately adding overlap (some families share a payment method too; rings
   vary which identifier they share, simulating operators with different caution
   levels). See [notebook 01](notebooks/01_data_synthesis.ipynb).

2. **A cost-blind "optimal" threshold wanted to flag 80% of all returns.**
   Minimizing pure dollar cost with no other constraint is mathematically
   correct and operationally absurd — no fraud-ops team can review 80% of
   returns. Fixed by adding an explicit review-capacity constraint (max 20% flag
   rate) and re-optimizing within it. See
   [notebook 03](notebooks/03_model_training_evaluation.ipynb).

3. **XGBoost's probabilities were badly overconfident.** `scale_pos_weight`
   (needed for class imbalance) gives good *ranking* but distorts raw
   probabilities — a "70% risk" order was actually abusive only ~16% of the time
   in the worst bin. Caught by checking a calibration curve, not just AUC. Fixed
   with isotonic calibration on a proper held-out slice of the training window
   (never touching the test set). Calibration gap: 0.300 → 0.046. See
   [notebook 03](notebooks/03_model_training_evaluation.ipynb).

4. **A degenerate bootstrap confidence interval.** Bootstrapping the ring
   detector's ~50-point test set gave `[1.000, 1.000]` — suspiciously tight.
   Turned out resampling a fixed, error-free test set can never introduce a new
   error, no matter how many times you resample it; the interval was measuring
   the wrong question. Fixed with repeated random train/test splits instead,
   which actually vary which clusters the model has never seen. See
   [notebook 04](notebooks/04_abuse_ring_detector.ipynb).

5. **We nearly shipped bug #2 and #3 in the wrong order.** While building
   notebook 03, we first computed the cost-optimal threshold on *raw* probabilities
   and only calibrated afterward — a scale mismatch, since isotonic calibration
   remaps the probability axis. Caught because the bootstrap numbers didn't match
   our known-correct pipeline output; fixed by calibrating first, then threshold-
   selecting on the calibrated scale, which is also how the actual deployed
   pipeline (`model/train_model.py`) works.

6. **A three-round Vercel deployment fight, eventually abandoned by design, not
   defeat.** We initially built this to be hosted, and hit real platform limits:
   XGBoost + SHAP's transitive dependencies (`shap` pulls in `numba`, which
   bundles a full LLVM via `llvmlite`) blew past Vercel's Python function size
   limit — 889MB, then 615MB, then 627MB across three attempted fixes, each
   teaching us something wrong about our own assumptions (a stale per-function
   `requirements.txt` that Vercel never actually reads; a platform-specific
   XGBoost binary difference we couldn't fully explain). We ultimately confirmed
   hosting isn't required for submission and redirected that effort into the
   local-first setup and this documentation instead — the honest engineering call
   given the actual constraint, not a workaround. The API still runs with XGBoost's
   *native* `pred_contribs` instead of the `shap` library for explanations (same
   underlying Tree-SHAP algorithm, zero extra dependencies) — a genuine
   improvement that survived the detour.

7. **Cross-platform pickle deserialization failure.** A model trained and
   pickled on Linux failed to load on Windows with `XGBoostError: input stream
   corrupted`, despite byte-identical files and matching library versions —
   a platform-level binary incompatibility, not corruption. Resolved by
   retraining locally on the target platform rather than transferring the
   artifact across platforms.

---

## Cost model (stated assumptions, not hidden constants)

- **False positive** (flagging a genuine return for review): a flat **₹150**,
  standing in for support-ticket handling cost + estimated customer-goodwill risk.
- **False negative** (missing a genuinely abusive return): the **order's value**
  — exposure scales with order size, not a flat number.
- **Review capacity constraint**: max **20%** of returns can realistically be
  sent to manual review — without this, the pure cost-minimum wants to flag 80%
  of all returns (see bug #2 above).

Swap in real numbers for a production deployment; the machinery (sweep + argmin,
subject to a capacity constraint) doesn't change. The interactive **cost-sensitivity
explorer** in the Metrics tab lets you see how the deployed threshold moves across
a grid of 72 real (not simulated) re-optimizations at different cost/capacity
assumptions.

## Limitations

- All results describe recovery of a synthetic generative rule, not validated
  real-world fraud rates.
- The ring detector's near-perfect metrics come from ~50 test clusters — read the
  repeated-split analysis, not the raw AUC, as the honest number.
- The fairness/bias check (in `model/calibration_and_fairness.py`) covers
  operational attributes (payment method, category, account tenure) only — this
  synthetic dataset has no demographic attributes, so it is **not** a
  protected-class fairness audit.
- Real fraud rings adapt to evade known signals over time; this is a static
  snapshot and would need periodic retraining against real labeled data in
  production.

## Defense-only scope

Every component here detects, scores, or flags for human/system review. Nothing
in this repository generates fraudulent content, automates an attack, or provides
offense capability of any kind — consistent with the track's disqualification
rule for offense-capable submissions.

---

## Architecture

```
riskguard/
├── data/                    Synthetic data generator + generated CSVs
├── model/                   Training scripts, evaluation, saved model artifacts
├── notebooks/               01-04: the full methodology, executed, with the
│                            debugging narrative above baked in as markdown
├── api/                     FastAPI backend (return-risk scoring, ring checking,
│                            metrics, sample clusters)
├── frontend/                React + TypeScript + Tailwind v4 dashboard
├── plots/                   Evaluation charts (PR/ROC, cost curve, calibration,
│                            SHAP summary, fairness breakdowns)
├── requirements.txt         Lean, deployment-only dependencies
└── requirements-training.txt   Extra deps for training/notebooks only (matplotlib,
                             networkx, shap)
```

**Frontend design credit:** built with [bklit-ui](https://github.com/bklit/bklit-ui)'s
`RingChart` (the risk gauge) and pattern system (the diagonal hatch texture on
unfilled gauge/track areas), and [kokonutui](https://github.com/kokonut-labs/kokonutui)'s
liquid-glass SVG displacement filter and sliding-pill tab navigation pattern. Both
are MIT-licensed component registries; we vendor the specific pieces we use directly
in `frontend/src/components/` since neither registry's CDN was reachable from our
build environment. We also found and patched a real upstream bug in bklit-ui's
`Ring` component (a `React.memo`-wrapped component wasn't being recognized by
their own type-detection helper) — noted in `frontend/src/components/charts/ring-chart.tsx`.

---

## Running it locally

**Backend:**
```bash
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
```

**Frontend** (separate terminal):
```bash
cd frontend
npm install
npm run dev
```
Open the printed local URL (usually `http://localhost:5173`).

**Retraining from scratch** (regenerates data + all models):
```bash
pip install -r requirements.txt -r requirements-training.txt
python data/generate_data.py
python model/prepare_features.py
python model/train_model.py
python model/abuse_ring_detector.py
python model/cost_sensitivity_grid.py
python model/calibration_and_fairness.py
python model/bootstrap_confidence_intervals.py
python model/business_impact.py
# copy fresh model artifacts into the API's served directory:
cp model/deployed_model.pkl model/xgb_model.pkl model/ring_detector_model.pkl \
   model/cost_sensitivity_grid.json api/models/
```

Or work through the notebooks in order (`notebooks/01` → `04`) for the full
narrated version, including every bug and fix described above, live.

---

## What the app does

- **Score Order** — enter a return's details, get a real-time risk score from the
  actual calibrated model, the recommended action (approve / manual review /
  auto-decline), the top contributing factors (native XGBoost Tree-SHAP), and a
  plain-English summary.
- **Abuse Ring** — pick a real sample cluster from training data (mix of true
  rings and benign sharing, ground truth hidden until you check), get the ring
  score, verdict, and factors.
- **Metrics** — the validated numbers above, the baseline-vs-ML comparison table,
  business-impact projections, and the interactive cost-sensitivity explorer.
