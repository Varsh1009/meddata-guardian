"""
FIDES Stage 3 — Four-Condition CDS (Comprehensive Dataset Sufficiency) Assessment

Evaluates whether a dataset is sufficiently rich for fair AI model training by
scoring four orthogonal sufficiency conditions:

    C1 — Pathway Sufficiency       (causal mediation, illegitimate path effects)
    C2 — Statistical Sufficiency   (sample size vs. required power per subgroup)
    C3 — Phenotypic Coverage       (distributional overlap with clinical manifold)
    C4 — Intersectional Sufficiency (adequate cell counts for all intersections)

The final CDS score is a weighted combination defined by spec.fairness_weights.
Thresholds: 0.75 for research / IRB audit, 0.85 for clinical deployment / FDA.
"""

from __future__ import annotations

import math
import re
import time
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from src.utils.research_spec import ResearchSpec
from src.utils.causal_discovery import CausalDiscoveryResult


# ── Threshold registry ────────────────────────────────────────────────────────

_CDS_THRESHOLDS: Dict[str, float] = {
    "research":            0.75,
    "irb_audit":           0.75,
    "clinical_deployment": 0.85,
    "fda_submission":      0.85,
}

# Minimum viable intersectional cell size (FDA guidance § 2.3)
_MIN_CELL_SIZE: int = 50

# KDE coverage density threshold (relative to full-dataset peak)
_KDE_DENSITY_THRESHOLD_QUANTILE: float = 0.05   # 5th-percentile of manifold densities


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class CDSResult:
    """Structured output of Stage 3 Four-Condition CDS Assessment."""

    cds_score: float
    """Weighted CDS score ∈ [0, 1]."""

    condition_scores: Dict[str, float]
    """Individual scores for 'pathway', 'statistical', 'coverage', 'intersectional'."""

    condition_details: Dict[str, dict]
    """Per-condition detailed breakdown (subgroup stats, path effects, etc.)."""

    insufficiency_masking_flags: List[str]
    """Warning strings raised when statistical power < 0.80 for any subgroup."""

    confidence_interval: Tuple[float, float]
    """95% CI around cds_score, width determined by minimum subgroup n."""

    threshold_met: bool
    """True when cds_score >= use-case threshold (0.75 research, 0.85 FDA)."""

    recommendations: List[str]
    """Actionable improvement suggestions when conditions are not fully met."""


# ── Utility helpers ───────────────────────────────────────────────────────────

def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _encode_series_to_numeric(series: pd.Series) -> np.ndarray:
    """Encode a categorical Series to numeric integers for regression."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(series.median()).values.astype(float)
    le = LabelEncoder()
    return le.fit_transform(series.fillna("__MISSING__").astype(str)).astype(float)


def _find_column_ci(col_norm: str, df_cols: List[str]) -> Optional[str]:
    """
    Case-insensitive, underscore-normalised column lookup.
    Returns the first matching column name in df_cols, or None.
    """
    norm = col_norm.lower().replace(" ", "_")
    for col in df_cols:
        if col.lower().replace(" ", "_") == norm:
            return col
        if norm in col.lower().replace(" ", "_"):
            return col
    return None


def _estimate_effect_size(y: np.ndarray, group_mask: np.ndarray) -> float:
    """Cohen's d between group-in and group-out on outcome y."""
    g1 = y[group_mask]
    g0 = y[~group_mask]
    if len(g1) < 2 or len(g0) < 2:
        return 0.0
    pooled_std = math.sqrt(
        (g1.var(ddof=1) * (len(g1) - 1) + g0.var(ddof=1) * (len(g0) - 1))
        / (len(g1) + len(g0) - 2 + 1e-9)
    )
    return abs(g1.mean() - g0.mean()) / (pooled_std + 1e-9)


# ── Condition 1 — Pathway Sufficiency ────────────────────────────────────────

def _condition1_pathway_sufficiency(
    df: pd.DataFrame,
    spec: ResearchSpec,
    causal_result: CausalDiscoveryResult,
) -> Tuple[float, dict, List[str]]:
    """
    For each protected attribute A, estimate:
        PSE_total        = total effect of A on target_variable
        PSE_illegitimate = sum of effects routed through illegitimate mediators

    Both are estimated via linear regression with and without blocking mediators
    (Baron–Kenny / potential-outcomes mediation approach).

    PS = 1 - |PSE_illegitimate| / (|PSE_total| + ε)
    """
    target = spec.target_variable
    if target not in df.columns:
        return 0.0, {"error": f"Target '{target}' not in DataFrame"}, []

    y = _encode_series_to_numeric(df[target])
    details: Dict[str, dict] = {}
    flags: List[str] = []
    ps_scores: List[float] = []

    # Build set of illegitimate mediator column names (normalised)
    illegit_meds: set = set()
    for path_obj in spec.illegitimate_paths:
        parts = [p.strip() for p in path_obj.path.replace("→", "|").split("|")]
        # Middle nodes are mediators (skip first and last)
        for med_raw in parts[1:-1]:
            col_found = _find_column_ci(med_raw, list(df.columns))
            if col_found:
                illegit_meds.add(col_found)

    dag = causal_result.dag

    for attr_spec in spec.protected_attributes:
        attr = attr_spec.attribute
        col = _find_column_ci(attr, list(df.columns))
        if col is None:
            details[attr] = {"error": "column not found in DataFrame"}
            continue

        A = _encode_series_to_numeric(df[col])

        # ── Total effect of A on Y (unconditional linear regression) ──────
        X_total = A.reshape(-1, 1)
        lr_total = LinearRegression().fit(X_total, y)
        pse_total = lr_total.coef_[0]

        # ── Effect through illegitimate paths (mediation) ──────────────────
        # Identify mediators: columns that lie on a directed path A → M → Y
        # in the DAG and are members of illegit_meds
        active_meds: List[str] = []
        for med_col in illegit_meds:
            if med_col in dag and col in dag:
                try:
                    if med_col in dag.successors(col) or any(
                        med_col in dag.successors(pred) for pred in dag.predecessors(med_col)
                    ):
                        active_meds.append(med_col)
                except Exception:
                    pass
            elif med_col in df.columns:
                active_meds.append(med_col)

        # PSE_illegitimate: effect of A on Y with mediators controlled
        if active_meds:
            med_arrays = [
                _encode_series_to_numeric(df[m]).reshape(-1, 1)
                for m in active_meds
            ]
            X_blocked = np.hstack([A.reshape(-1, 1)] + med_arrays)
            lr_blocked = LinearRegression().fit(X_blocked, y)
            pse_direct = lr_blocked.coef_[0]
            pse_illegitimate = pse_total - pse_direct
        else:
            pse_illegitimate = 0.0

        # ── Bootstrap 95% CI on PSE_illegitimate ──────────────────────────
        n = len(y)
        rng = np.random.RandomState(42)
        boot_pse_illgit: List[float] = []
        for _ in range(100):
            idx = rng.choice(n, size=n, replace=True)
            y_b = y[idx]
            A_b = A[idx]
            if active_meds:
                med_b = np.hstack([
                    _encode_series_to_numeric(df[m]).reshape(-1, 1)[idx]
                    for m in active_meds
                ])
                X_b_blk = np.hstack([A_b.reshape(-1, 1), med_b])
                X_b_tot = A_b.reshape(-1, 1)
                lr_tot_b = LinearRegression().fit(X_b_tot, y_b)
                lr_blk_b = LinearRegression().fit(X_b_blk, y_b)
                boot_pse_illgit.append(lr_tot_b.coef_[0] - lr_blk_b.coef_[0])
            else:
                boot_pse_illgit.append(0.0)

        ci_lo = float(np.percentile(boot_pse_illgit, 2.5))
        ci_hi = float(np.percentile(boot_pse_illgit, 97.5))

        ps = float(np.clip(
            1.0 - abs(pse_illegitimate) / (abs(pse_total) + 1e-9),
            0.0, 1.0,
        ))
        ps_scores.append(ps)

        details[attr] = {
            "pse_total":          round(pse_total, 6),
            "pse_illegitimate":   round(pse_illegitimate, 6),
            "pse_illgit_ci_95":   (round(ci_lo, 6), round(ci_hi, 6)),
            "active_mediators":   active_meds,
            "ps_score":           round(ps, 4),
        }

    overall_ps = float(np.mean(ps_scores)) if ps_scores else 0.5
    return overall_ps, {"per_attribute": details}, flags


# ── Condition 2 — Statistical Sufficiency ────────────────────────────────────

def _condition2_statistical_sufficiency(
    df: pd.DataFrame,
    spec: ResearchSpec,
) -> Tuple[float, dict, List[str]]:
    """
    For each subgroup G of each protected attribute:

        n_star_G = max(200, 10 * sqrt(n_features))
        Power(G) = Phi(sqrt(n_G) * d / sigma - 1.96)
            where d = Cohen's d vs. complement, sigma = 1 (standardised units)

    SS_G = min(1.0, n_G / n_star_G) * Power(G)
    SS   = mean over all G

    Raises an INSUFFICIENCY MASKING FLAG when Power(G) < 0.80.
    """
    target = spec.target_variable
    n_features = len(df.columns)
    n_star = max(200.0, 10.0 * math.sqrt(n_features))

    y = (
        _encode_series_to_numeric(df[target])
        if target in df.columns
        else np.zeros(len(df))
    )

    details: Dict[str, dict] = {}
    flags: List[str] = []
    ss_scores: List[float] = []

    all_specs = list(spec.protected_attributes) + list(spec.proxy_variables)

    for attr_spec in all_specs:
        attr = attr_spec.attribute
        col = _find_column_ci(attr, list(df.columns))
        if col is None:
            continue

        col_series = df[col]
        # Determine unique subgroups (up to 20 to avoid explosion)
        unique_vals = col_series.dropna().unique()
        if len(unique_vals) > 20:
            # Bin numeric columns into quartiles
            try:
                binned = pd.qcut(col_series, q=4, duplicates="drop")
                unique_vals = binned.dropna().unique()
                col_series = binned
            except Exception:
                unique_vals = unique_vals[:20]

        attr_detail: Dict[str, dict] = {}
        for g_val in unique_vals:
            mask = col_series == g_val
            n_g = int(mask.sum())
            if n_g == 0:
                continue

            d_g = _estimate_effect_size(y, mask.values)
            # Statistical power (one-sample approximation, sigma=1)
            sigma = float(y.std(ddof=1)) if y.std(ddof=1) > 0 else 1.0
            z_val = math.sqrt(n_g) * d_g / sigma - 1.96
            power_g = float(sp_stats.norm.cdf(z_val))
            power_g = max(power_g, 0.0)

            size_ratio = min(1.0, n_g / n_star)
            ss_g = size_ratio * power_g
            ss_scores.append(ss_g)

            if power_g < 0.80:
                msg = (
                    f"INSUFFICIENCY MASKING: subgroup '{attr}={g_val}' "
                    f"has Power={power_g:.3f} < 0.80 "
                    f"(n={n_g}, n*={int(n_star)}, d={d_g:.3f}). "
                    "Disparities may be undetectable — subgroup is under-powered."
                )
                flags.append(msg)

            attr_detail[str(g_val)] = {
                "n":           n_g,
                "n_star":      int(n_star),
                "cohen_d":     round(d_g, 4),
                "power":       round(power_g, 4),
                "size_ratio":  round(size_ratio, 4),
                "ss_g":        round(ss_g, 4),
            }

        details[attr] = attr_detail

    overall_ss = float(np.mean(ss_scores)) if ss_scores else 0.0
    return overall_ss, {"n_star": round(n_star, 2), "per_attribute": details}, flags


# ── Condition 3 — Phenotypic Coverage ────────────────────────────────────────

def _condition3_phenotypic_coverage(
    df: pd.DataFrame,
    spec: ResearchSpec,
    n_manifold_points: int = 500,
    random_state: int = 42,
) -> Tuple[float, dict, List[str]]:
    """
    Estimates distributional coverage of each subgroup over the clinical manifold.

    1. Build a KDE over numeric features of the *full* dataset.
    2. Sample n_manifold_points from it as the "clinical manifold".
    3. For each subgroup, build a KDE over the same feature space.
    4. Evaluate subgroup KDE at each manifold point.
    5. Coverage = fraction of manifold points with density > threshold
       (threshold = 5th-percentile of full-dataset KDE evaluated on manifold).
    """
    num_cols = _numeric_columns(df)
    # Drop target variable from feature space (we assess covariate coverage)
    target = spec.target_variable
    num_cols = [c for c in num_cols if c != target]

    flags: List[str] = []
    details: Dict[str, dict] = {}

    if len(num_cols) < 2:
        # Not enough numeric features for KDE — return neutral score
        return 0.5, {"note": "Fewer than 2 numeric features; coverage set to 0.5"}, flags

    rng = np.random.RandomState(random_state)

    # ── Full-dataset KDE ─────────────────────────────────────────────────────
    feat_matrix = df[num_cols].dropna().values.T   # shape (n_features, n_samples)
    if feat_matrix.shape[1] < 10:
        return 0.5, {"note": "Too few complete rows for KDE"}, flags

    try:
        kde_full = sp_stats.gaussian_kde(feat_matrix)
    except Exception as e:
        return 0.5, {"error": f"Full KDE failed: {e}"}, flags

    # ── Sample clinical manifold ─────────────────────────────────────────────
    manifold_pts = kde_full.resample(n_manifold_points, seed=random_state)
    # shape: (n_features, n_manifold_points)

    full_densities = kde_full(manifold_pts)
    density_threshold = float(np.percentile(full_densities, _KDE_DENSITY_THRESHOLD_QUANTILE * 100))

    cov_scores: List[float] = []

    all_attr_specs = list(spec.protected_attributes) + list(spec.proxy_variables)
    for attr_spec in all_attr_specs:
        attr = attr_spec.attribute
        col = _find_column_ci(attr, list(df.columns))
        if col is None:
            continue

        col_series = df[col]
        unique_vals = col_series.dropna().unique()
        if len(unique_vals) > 20:
            try:
                col_series = pd.qcut(df[col], q=4, duplicates="drop")
                unique_vals = col_series.dropna().unique()
            except Exception:
                unique_vals = unique_vals[:10]

        attr_detail: Dict[str, dict] = {}
        for g_val in unique_vals:
            mask = col_series == g_val
            sub_df = df.loc[mask, num_cols].dropna()

            if len(sub_df) < 5:
                attr_detail[str(g_val)] = {"coverage": 0.0, "n": len(sub_df),
                                            "note": "Too few samples for KDE"}
                cov_scores.append(0.0)
                continue

            try:
                kde_sub = sp_stats.gaussian_kde(sub_df.values.T)
                sub_densities = kde_sub(manifold_pts)
                coverage = float(np.mean(sub_densities > density_threshold))
            except Exception as kde_err:
                coverage = 0.0
                attr_detail[str(g_val)] = {
                    "coverage": 0.0, "n": len(sub_df),
                    "note": f"KDE failed: {kde_err}"
                }
                cov_scores.append(0.0)
                continue

            attr_detail[str(g_val)] = {
                "n":         len(sub_df),
                "coverage":  round(coverage, 4),
            }
            cov_scores.append(coverage)

        details[attr] = attr_detail

    overall_cov = float(np.mean(cov_scores)) if cov_scores else 0.5
    return (
        overall_cov,
        {
            "numeric_features_used": num_cols,
            "manifold_points":       n_manifold_points,
            "density_threshold":     round(density_threshold, 8),
            "per_attribute":         details,
        },
        flags,
    )


# ── Condition 4 — Intersectional Sufficiency ─────────────────────────────────

def _condition4_intersectional_sufficiency(
    df: pd.DataFrame,
    spec: ResearchSpec,
    min_cell_size: int = _MIN_CELL_SIZE,
) -> Tuple[float, dict, List[str]]:
    """
    For each intersection string in spec.relevant_intersections (e.g.
    "sex × race × age_group"), parse the constituent attribute names,
    locate them in df, and count samples in every cross-product cell.

    IS = fraction of cells with n >= min_cell_size.
    """
    flags: List[str] = []
    details: Dict[str, dict] = {}
    all_cell_results: List[bool] = []

    for intersection_str in spec.relevant_intersections:
        # Parse "A × B × C" — tolerates both '×' and 'x' as separator
        raw_parts = re.split(r"[×x]", intersection_str)
        col_names_norm = [p.strip() for p in raw_parts]

        # Find each component column in df
        resolved: List[Optional[str]] = []
        for cn in col_names_norm:
            found = _find_column_ci(cn, list(df.columns))
            resolved.append(found)

        missing = [col_names_norm[i] for i, r in enumerate(resolved) if r is None]
        if missing:
            details[intersection_str] = {
                "error": f"Columns not found in DataFrame: {missing}",
                "cells": {},
            }
            continue

        # Drop rows with NaN in any component column
        cols_present = [r for r in resolved if r is not None]
        sub = df[cols_present].dropna()

        if len(sub) == 0:
            details[intersection_str] = {"error": "No complete rows", "cells": {}}
            continue

        # Group by all component columns and count
        cell_counts = sub.groupby(cols_present).size()
        cell_details: Dict[str, int] = {}
        for cell_key, cnt in cell_counts.items():
            if isinstance(cell_key, tuple):
                key_str = " × ".join(str(v) for v in cell_key)
            else:
                key_str = str(cell_key)
            cell_details[key_str] = int(cnt)
            all_cell_results.append(cnt >= min_cell_size)

        n_cells = len(cell_details)
        n_sufficient = sum(1 for c in cell_details.values() if c >= min_cell_size)
        details[intersection_str] = {
            "columns_resolved": cols_present,
            "n_cells":          n_cells,
            "n_sufficient":     n_sufficient,
            "min_cell_size":    min_cell_size,
            "cells":            cell_details,
        }

    if not all_cell_results:
        return 0.5, {"note": "No intersection columns resolvable from DataFrame"}, flags

    is_score = float(np.mean(all_cell_results))
    return is_score, {"per_intersection": details}, flags


# ── Confidence interval helper ────────────────────────────────────────────────

def _compute_ci(
    cds_score: float,
    df: pd.DataFrame,
    spec: ResearchSpec,
) -> Tuple[float, float]:
    """
    Approximate 95% CI around cds_score.
    Width is inversely proportional to sqrt of the minimum subgroup size:
        half_width ≈ 1.96 * sqrt(p*(1-p) / n_min)  where p = cds_score
    """
    min_n = len(df)  # fallback
    for attr_spec in spec.protected_attributes:
        col = _find_column_ci(attr_spec.attribute, list(df.columns))
        if col:
            g_sizes = df[col].value_counts()
            if not g_sizes.empty:
                min_n = min(min_n, int(g_sizes.min()))

    p = float(np.clip(cds_score, 0.01, 0.99))
    half_width = 1.96 * math.sqrt(p * (1.0 - p) / max(min_n, 1))
    lo = float(np.clip(cds_score - half_width, 0.0, 1.0))
    hi = float(np.clip(cds_score + half_width, 0.0, 1.0))
    return lo, hi


# ── Recommendations generator ─────────────────────────────────────────────────

def _generate_recommendations(
    scores: Dict[str, float],
    details: Dict[str, dict],
    flags: List[str],
    spec: ResearchSpec,
) -> List[str]:
    recs: List[str] = []

    if scores.get("pathway", 1.0) < 0.70:
        recs.append(
            "PATHWAY: Illegitimate path effects are large. "
            "Collect additional data or apply causal deconfounding before training."
        )

    if scores.get("statistical", 1.0) < 0.70:
        recs.append(
            "STATISTICAL: Several subgroups are underpowered. "
            "Oversample minority subgroups to reach n* = "
            f"{details.get('statistical', {}).get('n_star', 200):.0f} per group."
        )

    if scores.get("coverage", 1.0) < 0.60:
        recs.append(
            "COVERAGE: Subgroups do not span the full clinical feature space. "
            "Augment with synthetic data (FIDES Stage 1 synthetic generator) "
            "calibrated to underrepresented regions of the manifold."
        )

    if scores.get("intersectional", 1.0) < 0.75:
        recs.append(
            f"INTERSECTIONAL: Many intersection cells have n < {_MIN_CELL_SIZE}. "
            "Consider stratified recruitment or synthetic augmentation for sparse cells."
        )

    if flags:
        recs.append(
            f"INSUFFICIENCY MASKING: {len(flags)} subgroup(s) are statistically "
            "underpowered to detect disparities. Report these gaps in your "
            "Model Card and obtain expert review before deployment."
        )

    if not recs:
        recs.append(
            "All four CDS conditions are within acceptable bounds. "
            "Proceed to Stage 4 (bias mitigation) following the fairness protocol."
        )

    return recs


# ── Main assessor class ───────────────────────────────────────────────────────

class CDSAssessor:
    """
    FIDES Stage 3 — Four-Condition CDS Assessor.

    Usage
    -----
    assessor = CDSAssessor(df, spec, causal_result)
    result   = assessor.assess()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        spec: ResearchSpec,
        causal_result: CausalDiscoveryResult,
        random_state: int = 42,
    ):
        self.df = df.copy()
        self.spec = spec
        self.causal_result = causal_result
        self.random_state = random_state

    # ── Individual condition methods ──────────────────────────────────────────

    def condition1_pathway_sufficiency(self) -> Tuple[float, dict, List[str]]:
        """
        Pathway Sufficiency (PS).

        Decomposes the total causal effect of each protected attribute on the
        target into legitimate vs. illegitimate path contributions using linear
        mediation analysis.  Bootstrap (100 iterations) provides 95% CI on
        PSE_illegitimate.

        Returns: (score, details_dict, flags)
        """
        print("  [C1] Computing pathway sufficiency via mediation analysis…")
        score, details, flags = _condition1_pathway_sufficiency(
            self.df, self.spec, self.causal_result
        )
        print(f"       C1 score = {score:.4f}")
        return score, details, flags

    def condition2_statistical_sufficiency(self) -> Tuple[float, dict, List[str]]:
        """
        Statistical Sufficiency (SS).

        Computes n*(G) = max(200, 10*sqrt(n_features)) for each subgroup G
        and statistical power using a normal-approximation power formula.
        Sets an INSUFFICIENCY MASKING FLAG when Power(G) < 0.80.

        Returns: (score, details_dict, flags)
        """
        print("  [C2] Computing statistical sufficiency and power per subgroup…")
        score, details, flags = _condition2_statistical_sufficiency(
            self.df, self.spec
        )
        print(f"       C2 score = {score:.4f}  |  "
              f"{len(flags)} insufficiency masking flag(s)")
        return score, details, flags

    def condition3_phenotypic_coverage(self) -> Tuple[float, dict, List[str]]:
        """
        Phenotypic Coverage (Cov).

        Estimates distributional overlap between each subgroup and the full
        clinical manifold using Gaussian KDE.  500 manifold samples are drawn
        from the full-dataset KDE; coverage = fraction with density > threshold.

        Returns: (score, details_dict, flags)
        """
        print("  [C3] Estimating phenotypic coverage via KDE…")
        score, details, flags = _condition3_phenotypic_coverage(
            self.df, self.spec, random_state=self.random_state
        )
        print(f"       C3 score = {score:.4f}")
        return score, details, flags

    def condition4_intersectional_sufficiency(self) -> Tuple[float, dict, List[str]]:
        """
        Intersectional Sufficiency (IS).

        Enumerates all cross-product cells from spec.relevant_intersections
        and reports the fraction with n >= 50 (minimum viable cell size).

        Returns: (score, details_dict, flags)
        """
        print("  [C4] Counting intersectional subgroup cells…")
        score, details, flags = _condition4_intersectional_sufficiency(
            self.df, self.spec
        )
        print(f"       C4 score = {score:.4f}")
        return score, details, flags

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def assess(self) -> CDSResult:
        """
        Run all four conditions and return a CDSResult.

        Weighted combination uses spec.fairness_weights:
            CDS = w_pathway * C1 + w_statistical * C2
                + w_coverage * C3 + w_intersectional * C4
        """
        t_start = time.time()
        print("=" * 64)
        print("FIDES Stage 3 — Four-Condition CDS Assessment")
        print(f"  Domain       : {self.spec.domain}")
        print(f"  Use case     : {self.spec.use_case}")
        print(f"  Dataset      : {self.df.shape[0]} rows × {self.df.shape[1]} cols")
        print(f"  Weights      : {self.spec.fairness_weights}")
        print("=" * 64)

        all_flags: List[str] = []
        all_details: Dict[str, dict] = {}

        print("\n── Condition 1: Pathway Sufficiency ──────────────────────────")
        c1_score, c1_det, c1_flags = self.condition1_pathway_sufficiency()
        all_flags.extend(c1_flags)
        all_details["pathway"] = c1_det

        print("\n── Condition 2: Statistical Sufficiency ──────────────────────")
        c2_score, c2_det, c2_flags = self.condition2_statistical_sufficiency()
        all_flags.extend(c2_flags)
        all_details["statistical"] = c2_det

        print("\n── Condition 3: Phenotypic Coverage ──────────────────────────")
        c3_score, c3_det, c3_flags = self.condition3_phenotypic_coverage()
        all_flags.extend(c3_flags)
        all_details["coverage"] = c3_det

        print("\n── Condition 4: Intersectional Sufficiency ───────────────────")
        c4_score, c4_det, c4_flags = self.condition4_intersectional_sufficiency()
        all_flags.extend(c4_flags)
        all_details["intersectional"] = c4_det

        # ── Weighted CDS score ────────────────────────────────────────────
        w = self.spec.fairness_weights
        w_pathway       = w.get("pathway",       0.25)
        w_statistical   = w.get("statistical",   0.25)
        w_coverage      = w.get("coverage",      0.25)
        w_intersectional = w.get("intersectional", 0.25)

        # Renormalise in case weights don't sum to 1
        w_sum = w_pathway + w_statistical + w_coverage + w_intersectional
        if w_sum > 0:
            w_pathway        /= w_sum
            w_statistical    /= w_sum
            w_coverage       /= w_sum
            w_intersectional /= w_sum

        cds_score = (
            w_pathway       * c1_score
            + w_statistical   * c2_score
            + w_coverage      * c3_score
            + w_intersectional * c4_score
        )
        cds_score = float(np.clip(cds_score, 0.0, 1.0))

        condition_scores = {
            "pathway":        round(c1_score, 4),
            "statistical":    round(c2_score, 4),
            "coverage":       round(c3_score, 4),
            "intersectional": round(c4_score, 4),
        }

        # ── Confidence interval ───────────────────────────────────────────
        ci = _compute_ci(cds_score, self.df, self.spec)

        # ── Threshold ────────────────────────────────────────────────────
        threshold = _CDS_THRESHOLDS.get(self.spec.use_case, 0.75)
        threshold_met = cds_score >= threshold

        # ── Recommendations ───────────────────────────────────────────────
        recs = _generate_recommendations(
            condition_scores, all_details, all_flags, self.spec
        )

        elapsed = time.time() - t_start

        # ── Print summary ─────────────────────────────────────────────────
        print("\n" + "=" * 64)
        print("CDS ASSESSMENT SUMMARY")
        print("=" * 64)
        print(f"  C1 Pathway Sufficiency      : {c1_score:.4f}  (w={w_pathway:.2f})")
        print(f"  C2 Statistical Sufficiency  : {c2_score:.4f}  (w={w_statistical:.2f})")
        print(f"  C3 Phenotypic Coverage      : {c3_score:.4f}  (w={w_coverage:.2f})")
        print(f"  C4 Intersectional Suff.     : {c4_score:.4f}  (w={w_intersectional:.2f})")
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  CDS Score                   : {cds_score:.4f}  "
              f"(95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        print(f"  Threshold ({self.spec.use_case:>20s}) : {threshold}")
        status = "PASS" if threshold_met else "FAIL"
        print(f"  Decision                    : {status}")

        if all_flags:
            print(f"\n  Insufficiency Masking Flags ({len(all_flags)}):")
            for f_msg in all_flags:
                print(f"    WARNING: {f_msg}")

        if recs:
            print(f"\n  Recommendations:")
            for rec in recs:
                print(f"    • {rec}")

        print(f"\nAssessment complete in {elapsed:.2f}s")
        print("=" * 64)

        return CDSResult(
            cds_score=round(cds_score, 6),
            condition_scores=condition_scores,
            condition_details=all_details,
            insufficiency_masking_flags=all_flags,
            confidence_interval=ci,
            threshold_met=threshold_met,
            recommendations=recs,
        )


# ── Module-level convenience function ────────────────────────────────────────

def assess_dataset_sufficiency(
    df: pd.DataFrame,
    spec: ResearchSpec,
    causal_result: CausalDiscoveryResult,
    random_state: int = 42,
) -> CDSResult:
    """
    Convenience wrapper: instantiate CDSAssessor and run the full assessment.

    Parameters
    ----------
    df            : de-identified pandas DataFrame (Stage 1 output)
    spec          : ResearchSpec from Stage 0
    causal_result : CausalDiscoveryResult from Stage 2
    random_state  : reproducibility seed

    Returns
    -------
    CDSResult
    """
    assessor = CDSAssessor(df, spec, causal_result, random_state=random_state)
    return assessor.assess()


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.research_spec import build_research_spec
    from src.utils.causal_discovery import run_causal_discovery

    np.random.seed(42)
    n = 600

    age_arr      = np.random.normal(55, 12, n)
    sex_arr      = np.random.choice(["Male", "Female"], n, p=[0.55, 0.45])
    race_arr     = np.random.choice(["White", "Black", "Hispanic", "Asian"], n,
                                     p=[0.60, 0.13, 0.20, 0.07])
    insurance_arr = np.random.choice(["Private", "Medicare", "Medicaid", "None"], n,
                                      p=[0.50, 0.25, 0.15, 0.10])
    troponin_arr = np.clip(
        0.025 * age_arr
        + 0.3 * (sex_arr == "Male").astype(float)
        + np.random.normal(0, 0.4, n),
        0, 5,
    )
    pain_arr     = 4.5 + 0.6 * (sex_arr == "Male") + np.random.normal(0, 1.5, n)
    ttd_arr      = np.random.normal(3.5, 1.2, n)
    mi_arr       = (troponin_arr > 1.0).astype(int)

    df_demo = pd.DataFrame({
        "age":                  age_arr,
        "sex":                  sex_arr,
        "race":                 race_arr,
        "insurance":            insurance_arr,
        "troponin":             troponin_arr,
        "pain_score_recorded":  pain_arr,
        "time_to_diagnosis":    ttd_arr,
        "MI_outcome":           mi_arr,
    })

    spec = build_research_spec(
        domain="cardiology",
        target_variable="MI_outcome",
        target_type="binary",
        use_case="fda_submission",
        columns=list(df_demo.columns),
        intent="Predict MI risk in adult ED population",
    )

    print("\n=== Running Stage 2 first (required input for Stage 3) ===\n")
    causal_res = run_causal_discovery(df_demo, spec, epsilon=1.0)

    print("\n=== Running Stage 3 CDS Assessment ===\n")
    cds_res = assess_dataset_sufficiency(df_demo, spec, causal_res)

    print("\n─── CDSResult fields ───")
    print(f"  cds_score             : {cds_res.cds_score}")
    print(f"  condition_scores      : {cds_res.condition_scores}")
    print(f"  threshold_met         : {cds_res.threshold_met}")
    print(f"  confidence_interval   : {cds_res.confidence_interval}")
    print(f"  n_flags               : {len(cds_res.insufficiency_masking_flags)}")
    print(f"  n_recommendations     : {len(cds_res.recommendations)}")
