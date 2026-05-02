"""Lightweight, deterministic ML scorer for misrepresentation risk.

Sits next to the few-shot LLM classifier (`fraud_classifier.py`) and provides
a *second*, fully reproducible probability estimate that does not depend on
any external API. The two scores are then combined into an ensemble verdict
in the pipeline.

Why not call this "training" of a deep model? Because for the case-study
prototype the better engineering trade-off is:

1. Hand-curated training set of 9 landmark cases (5 frauds + 4 controls)
   from `fraud_classifier.FEW_SHOT_LIBRARY` — too small for a neural net.
2. Train a small *logistic regression* on engineered features that we can
   actually compute deterministically from the SEC XBRL pull (Beneish M
   components, Piotroski strength, Altman Z zone, AR/Revenue gap, NI/CFO
   gap, gross-margin volatility, goodwill/equity, anomaly count, LLM red-
   flag count).
3. Persist the fitted weights in this file so the scorer is portable and
   reviewable; no model file checked into git, no scikit-learn at runtime.
4. Evaluate calibration on the 9-case set as a sanity check (printed if the
   module is run directly: `python -m muddy_reporter.ml_scorer`).

For a real product you would replace step 1 with a labeled corpus
(Compustat × AAER) and a proper pipeline. The structure here is identical;
only the training data changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Hand-labeled training set (mirrors the few-shot LLM library).
# Numbers are illustrative based on publicly available analyses; they are NOT
# audited values and exist only to anchor the model's learned prior.
# ---------------------------------------------------------------------------
#
# Each row: (label, features_dict). label = 1 means "later confirmed
# misrepresentation", 0 means "control / clean".
#
# Features (all optional, missing = NaN handled at inference time):
#   beneish_m            : Beneish M-Score (>-1.78 = manipulator territory)
#   piotroski_f          : Piotroski F (0-9, lower = weaker)
#   altman_z             : Altman Z'-Score (<1.23 = distress)
#   ar_growth_minus_rev  : YoY (AR growth - revenue growth), proportion (0.30 = 30 pp gap)
#   ni_minus_cfo_ratio   : (Net income - CFO) / |CFO|, ratio
#   gross_margin_jump    : abs(YoY change in gross margin %), proportion
#   goodwill_to_equity   : goodwill / total equity
#   anomaly_count        : count of XBRL anomalies surfaced
#   llm_redflag_count    : count of LLM-extracted findings labeled fact|inference
#
# We deliberately keep the feature count small so a 9-row training set isn't
# laughable. With more labels, broaden it — the inference path here doesn't
# care about feature dimensionality.

_TRAIN: list[tuple[int, dict[str, float | None]]] = [
    # --- Confirmed frauds ---
    (1, {  # Enron (FY2000, prior to collapse)
        "beneish_m": 1.0, "piotroski_f": 3, "altman_z": 1.0,
        "ar_growth_minus_rev": 0.35, "ni_minus_cfo_ratio": 0.6,
        "gross_margin_jump": 0.08, "goodwill_to_equity": 0.4,
        "anomaly_count": 5, "llm_redflag_count": 6,
    }),
    (1, {  # Wirecard (FY2018)
        "beneish_m": 0.85, "piotroski_f": 4, "altman_z": 1.4,
        "ar_growth_minus_rev": 0.25, "ni_minus_cfo_ratio": 0.5,
        "gross_margin_jump": 0.05, "goodwill_to_equity": 0.55,
        "anomaly_count": 4, "llm_redflag_count": 5,
    }),
    (1, {  # Luckin Coffee (FY2019, pre-exposure)
        "beneish_m": 1.2, "piotroski_f": 3, "altman_z": None,
        "ar_growth_minus_rev": 0.55, "ni_minus_cfo_ratio": 0.7,
        "gross_margin_jump": 0.10, "goodwill_to_equity": 0.10,
        "anomaly_count": 6, "llm_redflag_count": 7,
    }),
    (1, {  # Nikola (FY2020)
        "beneish_m": None, "piotroski_f": 2, "altman_z": 0.8,
        "ar_growth_minus_rev": None, "ni_minus_cfo_ratio": 1.0,
        "gross_margin_jump": None, "goodwill_to_equity": 0.0,
        "anomaly_count": 3, "llm_redflag_count": 6,
    }),
    (1, {  # WorldCom (FY2001)
        "beneish_m": 0.9, "piotroski_f": 4, "altman_z": 1.2,
        "ar_growth_minus_rev": 0.18, "ni_minus_cfo_ratio": 0.4,
        "gross_margin_jump": 0.06, "goodwill_to_equity": 0.65,
        "anomaly_count": 4, "llm_redflag_count": 5,
    }),
    # --- Clean controls ---
    (0, {  # Apple
        "beneish_m": -2.5, "piotroski_f": 8, "altman_z": 1.2,  # Z low for buyback firms; non-distress
        "ar_growth_minus_rev": 0.02, "ni_minus_cfo_ratio": -0.10,
        "gross_margin_jump": 0.01, "goodwill_to_equity": 0.0,
        "anomaly_count": 0, "llm_redflag_count": 1,
    }),
    (0, {  # Costco
        "beneish_m": -2.7, "piotroski_f": 8, "altman_z": 4.0,
        "ar_growth_minus_rev": 0.0, "ni_minus_cfo_ratio": -0.15,
        "gross_margin_jump": 0.005, "goodwill_to_equity": 0.0,
        "anomaly_count": 0, "llm_redflag_count": 0,
    }),
    (0, {  # P&G
        "beneish_m": -2.6, "piotroski_f": 7, "altman_z": 3.0,
        "ar_growth_minus_rev": 0.01, "ni_minus_cfo_ratio": -0.10,
        "gross_margin_jump": 0.01, "goodwill_to_equity": 0.5,
        "anomaly_count": 0, "llm_redflag_count": 1,
    }),
    (0, {  # Microsoft
        "beneish_m": -2.7, "piotroski_f": 8, "altman_z": 3.5,
        "ar_growth_minus_rev": 0.02, "ni_minus_cfo_ratio": -0.20,
        "gross_margin_jump": 0.01, "goodwill_to_equity": 0.3,
        "anomaly_count": 0, "llm_redflag_count": 1,
    }),
]


_FEATURE_ORDER: list[str] = [
    "beneish_m",
    "piotroski_f",
    "altman_z",
    "ar_growth_minus_rev",
    "ni_minus_cfo_ratio",
    "gross_margin_jump",
    "goodwill_to_equity",
    "anomaly_count",
    "llm_redflag_count",
]


# ---------------------------------------------------------------------------
# Training: pure-Python logistic regression so we don't import sklearn at
# runtime. The model is tiny (9 features, 9 rows) so a few hundred GD
# iterations are more than enough.
# ---------------------------------------------------------------------------


@dataclass
class _StandardizedTrain:
    means: list[float]
    stds: list[float]
    X: list[list[float]]  # standardized
    mask: list[list[bool]]  # 1 if feature was present, 0 if imputed
    y: list[int]


def _standardize() -> _StandardizedTrain:
    n = len(_TRAIN)
    feats = _FEATURE_ORDER
    raw_cols: list[list[float | None]] = [[r[1].get(f) for r in _TRAIN] for f in feats]
    means: list[float] = []
    stds: list[float] = []
    for col in raw_cols:
        vals = [v for v in col if v is not None]
        m = sum(vals) / len(vals) if vals else 0.0
        var = sum((v - m) ** 2 for v in vals) / max(1, len(vals))
        s = math.sqrt(var) or 1.0
        means.append(m)
        stds.append(s)
    X: list[list[float]] = []
    mask: list[list[bool]] = []
    for i in range(n):
        row: list[float] = []
        m_row: list[bool] = []
        for j, f in enumerate(feats):
            v = raw_cols[j][i]
            if v is None:
                row.append(0.0)  # impute to mean (after standardization, that's 0)
                m_row.append(False)
            else:
                row.append((v - means[j]) / stds[j])
                m_row.append(True)
        X.append(row)
        mask.append(m_row)
    y = [r[0] for r in _TRAIN]
    return _StandardizedTrain(means=means, stds=stds, X=X, mask=mask, y=y)


def _sigmoid(z: float) -> float:
    if z < -50:
        return 0.0
    if z > 50:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _train_lr(*, lr: float = 0.25, iters: int = 1500, l2: float = 0.5) -> tuple[list[float], float]:
    """Train binary logistic regression with L2 regularization. Returns (weights, bias)."""
    data = _standardize()
    n_features = len(_FEATURE_ORDER)
    w = [0.0] * n_features
    b = 0.0
    for _ in range(iters):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for xi, mi, yi in zip(data.X, data.mask, data.y):
            z = b + sum(w[j] * xi[j] for j in range(n_features) if mi[j])
            p = _sigmoid(z)
            err = p - yi
            for j in range(n_features):
                if mi[j]:
                    grad_w[j] += err * xi[j]
            grad_b += err
        for j in range(n_features):
            grad_w[j] = grad_w[j] / len(data.y) + l2 * w[j]
        grad_b /= len(data.y)
        for j in range(n_features):
            w[j] -= lr * grad_w[j]
        b -= lr * grad_b
    return w, b


# Train once at import. Cheap (~5ms) and the result is deterministic.
_W, _B = _train_lr()
_TRAIN_NORM = _standardize()


# ---------------------------------------------------------------------------
# Inference + feature engineering from our pipeline payloads.
# ---------------------------------------------------------------------------


@dataclass
class MLScore:
    probability: float          # 0-1, raw LR output
    verdict: str                # clean | watch | suspicious | likely_misrepresentation
    confidence: str             # low | medium | high
    feature_contributions: list[dict[str, Any]] = field(default_factory=list)
    coverage: float = 0.0       # share of features that were observed (0-1)
    model: str = "logreg-9feat-v1"
    notes: list[str] = field(default_factory=list)


def _verdict_from_prob(p: float) -> str:
    if p >= 0.70:
        return "likely_misrepresentation"
    if p >= 0.45:
        return "suspicious"
    if p >= 0.20:
        return "watch"
    return "clean"


def _confidence_from_coverage(coverage: float) -> str:
    if coverage >= 0.75:
        return "high"
    if coverage >= 0.50:
        return "medium"
    return "low"


def _safe(d: dict[str, Any] | None, key: str) -> Any:
    if not d:
        return None
    v = d.get(key)
    return v if v is not None else None


def _compute_features(*, financial_brief: dict | None,
                      llm_findings_summary: dict[str, int]) -> dict[str, float | None]:
    """Translate our pipeline payloads into the same 9-feature vector used in training."""
    fs = (financial_brief or {}).get("forensic_scores") or {}
    table = (financial_brief or {}).get("annual_table") or []
    anomalies = (financial_brief or {}).get("anomalies") or []

    def _ratio(a, b):
        try:
            if a is None or b is None or b == 0:
                return None
            return float(a) / float(b)
        except (TypeError, ValueError):
            return None

    feat: dict[str, float | None] = {
        "beneish_m": fs.get("beneish_m"),
        "piotroski_f": fs.get("piotroski_f"),
        "altman_z": fs.get("altman_z"),
        "ar_growth_minus_rev": None,
        "ni_minus_cfo_ratio": None,
        "gross_margin_jump": None,
        "goodwill_to_equity": None,
        "anomaly_count": float(len(anomalies)),
        "llm_redflag_count": float(
            (llm_findings_summary.get("fact", 0) or 0)
            + (llm_findings_summary.get("inference", 0) or 0)
        ),
    }

    if len(table) >= 2:
        a, b = table[-2], table[-1]
        rev_g = _ratio((b.get("revenue") or 0) - (a.get("revenue") or 0), a.get("revenue"))
        ar_g = _ratio((b.get("accounts_receivable") or 0) - (a.get("accounts_receivable") or 0),
                      a.get("accounts_receivable"))
        if rev_g is not None and ar_g is not None:
            feat["ar_growth_minus_rev"] = ar_g - rev_g
        ni = b.get("net_income")
        cfo = b.get("operating_cash_flow")
        if ni is not None and cfo not in (None, 0):
            try:
                feat["ni_minus_cfo_ratio"] = (float(ni) - float(cfo)) / abs(float(cfo))
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        gp_a = a.get("gross_profit"); rev_a = a.get("revenue")
        gp_b = b.get("gross_profit"); rev_b = b.get("revenue")
        if all(v is not None and v != 0 for v in (gp_a, rev_a, gp_b, rev_b)):
            try:
                feat["gross_margin_jump"] = abs(
                    float(gp_b) / float(rev_b) - float(gp_a) / float(rev_a)
                )
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        gw = b.get("goodwill")
        eq = b.get("stockholders_equity") or b.get("total_equity")
        if gw is not None and eq not in (None, 0):
            try:
                feat["goodwill_to_equity"] = float(gw) / float(eq)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    return feat


def score(*, financial_brief: dict | None,
          llm_findings_summary: dict[str, int]) -> MLScore:
    feat = _compute_features(
        financial_brief=financial_brief,
        llm_findings_summary=llm_findings_summary,
    )
    contributions: list[dict[str, Any]] = []
    z = _B
    observed = 0
    total = len(_FEATURE_ORDER)
    for j, name in enumerate(_FEATURE_ORDER):
        v = feat.get(name)
        if v is None:
            contributions.append({
                "feature": name, "value": None, "z": None,
                "weight": round(_W[j], 3), "logit_contribution": 0.0,
                "observed": False,
            })
            continue
        observed += 1
        std_val = (v - _TRAIN_NORM.means[j]) / (_TRAIN_NORM.stds[j] or 1.0)
        contrib = _W[j] * std_val
        z += contrib
        contributions.append({
            "feature": name,
            "value": round(float(v), 4),
            "z": round(float(std_val), 3),
            "weight": round(_W[j], 3),
            "logit_contribution": round(float(contrib), 3),
            "observed": True,
        })

    p = _sigmoid(z)
    coverage = observed / total
    verdict = _verdict_from_prob(p)
    notes: list[str] = []
    if observed < 4:
        notes.append(
            "Sparse XBRL coverage (<4 features observed); the prior dominates so trust this score less."
        )
    if feat.get("beneish_m") is None and feat.get("altman_z") is None:
        notes.append(
            "Both Beneish M and Altman Z were unavailable — typical for foreign issuers (20-F) with thin XBRL tagging."
        )
    return MLScore(
        probability=round(float(p), 3),
        verdict=verdict,
        confidence=_confidence_from_coverage(coverage),
        feature_contributions=sorted(
            contributions, key=lambda c: abs(c["logit_contribution"]), reverse=True
        ),
        coverage=round(coverage, 2),
        notes=notes,
    )


def ensemble(llm_classifier: dict | None, ml: MLScore) -> dict[str, Any]:
    """Fuse the LLM few-shot classifier and the deterministic LR model.

    Both are calibrated probabilities of misrepresentation. We average them,
    weighted by each estimator's confidence (high=1.0, medium=0.7, low=0.4).
    Track per-source vote so the UI can display "models agree / disagree".
    """
    def _w(c: str | None) -> float:
        return {"high": 1.0, "medium": 0.7, "low": 0.4}.get((c or "low").lower(), 0.4)

    llm_prob = None
    llm_conf = "low"
    llm_verdict = None
    if llm_classifier:
        try:
            llm_prob = float(llm_classifier.get("fraud_probability"))
        except (TypeError, ValueError):
            llm_prob = None
        llm_conf = llm_classifier.get("confidence") or "low"
        llm_verdict = llm_classifier.get("verdict")

    ml_w = _w(ml.confidence)
    if llm_prob is None:
        combined = ml.probability
        agreement = "ml-only"
    else:
        llm_w = _w(llm_conf)
        combined = (llm_prob * llm_w + ml.probability * ml_w) / max(0.001, (llm_w + ml_w))
        # Verdict-level agreement: do both estimators land in the same bucket?
        a = (llm_verdict or "watch").lower().replace("likely_fraud", "likely_misrepresentation")
        b = ml.verdict
        agreement = "agree" if a == b else "disagree"

    combined = round(min(1.0, max(0.0, combined)), 3)
    return {
        "combined_probability": combined,
        "verdict": _verdict_from_prob(combined),
        "agreement": agreement,
        "votes": {
            "llm_few_shot": {
                "probability": llm_prob, "verdict": llm_verdict, "confidence": llm_conf,
            },
            "ml_logreg": {
                "probability": ml.probability, "verdict": ml.verdict,
                "confidence": ml.confidence, "model": ml.model, "coverage": ml.coverage,
            },
        },
    }


# ---------------------------------------------------------------------------
# Self-check: print weights + leave-one-out accuracy on the 9-row training
# set when the module is run directly.
# ---------------------------------------------------------------------------


def _leave_one_out_accuracy() -> tuple[float, list[tuple[int, int, float]]]:
    global _TRAIN  # noqa: PLW0603 — intentional swap-and-restore for LOO

    original = _TRAIN
    correct = 0
    rows = []
    try:
        for i in range(len(original)):
            held = original[i]
            _TRAIN = original[:i] + original[i + 1:]
            w, bias = _train_lr()
            norm = _standardize()
            feat = held[1]
            z = bias
            for j, f in enumerate(_FEATURE_ORDER):
                v = feat.get(f)
                if v is None:
                    continue
                sv = (v - norm.means[j]) / (norm.stds[j] or 1.0)
                z += w[j] * sv
            p = _sigmoid(z)
            pred = 1 if p >= 0.5 else 0
            rows.append((held[0], pred, round(p, 3)))
            if pred == held[0]:
                correct += 1
    finally:
        _TRAIN = original
    return correct / len(original), rows


if __name__ == "__main__":  # pragma: no cover
    print("Trained LR weights (in standardized space):")
    for f, w in zip(_FEATURE_ORDER, _W):
        print(f"  {f:<22} {w:+.3f}")
    print(f"  {'(intercept)':<22} {_B:+.3f}\n")
    acc, preds = _leave_one_out_accuracy()
    print(f"Leave-one-out accuracy on 9-case calibration set: {acc * 100:.0f}%")
    for true, pred, prob in preds:
        mark = "✓" if true == pred else "✗"
        print(f"  {mark} true={true}  pred={pred}  p={prob}")
