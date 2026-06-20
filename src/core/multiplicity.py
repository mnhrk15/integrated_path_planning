"""Multiple-comparison corrections for the RQ statistical ledger.

Pure functions over p-value sequences. The axis-A review (``docs/CODE_REVIEW_
axisA_20260619.md`` point 8) flagged that the RQ suite reports many significance
tests -- RQ2 pooled closest/onset KS, RQ1b per-scenario single-vs-robust Fisher
-- without any family-wise / false-discovery management, while simultaneously
calling non-significant results "indistinguishable". That asymmetric rigor (only
the significant side escapes multiplicity) is what this module removes.

We report BOTH corrections, never one alone:
  * Benjamini-Hochberg FDR (primary): controls the expected proportion of false
    discoveries among rejections; less conservative, appropriate when the family
    is a screen of related hypotheses.
  * Holm-Bonferroni (conservative sensitivity): controls the family-wise error
    rate (probability of ANY false rejection); a stricter bar a finding should
    still ideally clear.
So "survives correction" is honest under either definition.

NaNs are NOT hypotheses. A Fisher test on an empty arm or a KS on an empty pool
returns NaN -- there is no p-value to correct -- so NaN entries are excluded from
the family size rather than treated as p=1.0 (which would silently shrink every
other adjusted value). NaN in -> NaN out, and the family size m counts only the
finite p-values actually tested.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import false_discovery_control


def _finite_mask(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    return np.isfinite(p)


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (NaN-aware).

    Returns an array the same length as ``pvalues``. Non-finite inputs map to NaN
    and are excluded from the family size m (the number of finite p-values), so
    the correction is over the hypotheses actually tested, not padded with
    phantom NaN ones. Delegates the finite-subset arithmetic to scipy's
    ``false_discovery_control`` (method="bh") so the monotone step-up enforcement
    matches the reference implementation.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = _finite_mask(p)
    if not mask.any():
        return out
    out[mask] = false_discovery_control(p[mask], method="bh")
    return out


def holm_bonferroni(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down FWER-adjusted p-values (NaN-aware).

    Standard step-down: sort the m finite p-values ascending, scale the k-th
    smallest by (m - k + 1), enforce monotone non-decreasing along the sorted
    order, and clip to 1. Non-finite inputs map to NaN and do not count toward m.
    A finding is FWER-significant at level alpha iff its adjusted value <= alpha.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    idx = np.flatnonzero(_finite_mask(p))
    if idx.size == 0:
        return out
    m = idx.size
    finite = p[idx]
    order = np.argsort(finite, kind="mergesort")  # stable -> deterministic ties
    scaled = (m - np.arange(m)) * finite[order]    # (m-k+1) * p_(k), k=1..m
    # Step-down monotonicity: the adjusted value at rank k is the running maximum
    # of the scaled p-values up to k, then clipped to 1.
    adjusted_sorted = np.clip(np.maximum.accumulate(scaled), None, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    out[idx] = adjusted
    return out


def adjust(pvalues: Sequence[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Both corrections over ONE family of p-values, plus rejection flags.

    Returns a dict with ``bh_q`` / ``holm_p`` (adjusted values, NaN-aware) and
    ``bh_reject`` / ``holm_reject`` boolean arrays. A hypothesis is rejected when
    its adjusted value is ``<= alpha`` (the standard inclusive NHST/FDR boundary;
    a discrete Fisher/KS p landing exactly on alpha must reject, not be silently
    dropped -- boundary honesty is the whole point of this module). NaN -> False.
    ``m`` is the finite family size.
    """
    bh = benjamini_hochberg(pvalues)
    holm = holm_bonferroni(pvalues)
    return {
        "bh_q": bh,
        "holm_p": holm,
        "bh_reject": np.where(np.isfinite(bh), bh <= alpha, False),
        "holm_reject": np.where(np.isfinite(holm), holm <= alpha, False),
        "m": int(_finite_mask(pvalues).sum()),
    }


def build_ledger(tests: List[Dict], alpha: float = 0.05) -> List[Dict]:
    """Apply BH + Holm within each ``family`` AND study-wide, return ledger rows.

    ``tests`` is a list of dicts each carrying at least ``p_value`` and ``family``
    (plus any descriptive fields such as ``test_id``/``description``, passed
    through untouched). Two corrections are attached to every row:
      * within-family  : ``family_bh_q`` / ``family_holm_p`` / ``*_reject`` and
        ``family_size`` -- the correct unit when families answer distinct
        questions (RQ2 fidelity KS vs RQ1b planning-safety Fisher).
      * study-wide     : ``overall_bh_q`` / ``overall_holm_p`` / ``*_reject`` and
        ``overall_size`` -- the cross-cutting view the review (point 8) asks for.
    Reporting both means a "survives correction" claim cannot be gamed by a
    convenient family definition. Input order is preserved.
    """
    rows = [dict(t) for t in tests]
    if not rows:
        return rows

    # Study-wide correction over the whole family.
    overall_p = [r.get("p_value", np.nan) for r in rows]
    overall = adjust(overall_p, alpha)
    overall_m = overall["m"]
    for i, r in enumerate(rows):
        r["overall_bh_q"] = float(overall["bh_q"][i])
        r["overall_holm_p"] = float(overall["holm_p"][i])
        r["overall_bh_reject"] = bool(overall["bh_reject"][i])
        r["overall_holm_reject"] = bool(overall["holm_reject"][i])
        r["overall_size"] = overall_m

    # Within-family correction (group by the `family` label, order-stable).
    families: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        families.setdefault(str(r.get("family", "")), []).append(i)
    for fam, members in families.items():
        fam_p = [rows[i].get("p_value", np.nan) for i in members]
        fam = adjust(fam_p, alpha)
        for k, i in enumerate(members):
            rows[i]["family_bh_q"] = float(fam["bh_q"][k])
            rows[i]["family_holm_p"] = float(fam["holm_p"][k])
            rows[i]["family_bh_reject"] = bool(fam["bh_reject"][k])
            rows[i]["family_holm_reject"] = bool(fam["holm_reject"][k])
            rows[i]["family_size"] = fam["m"]
    return rows
