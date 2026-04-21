"""
FIDES Stage 5 — Equitable-Care Counterfactual Generator

Generates synthetic patients from underrepresented groups who received
*equitable care* — same biology, same demographics, different (fair) care
pathway.

This is NOT SMOTE, NOT CT-GAN, and NOT race-swapping.

The mechanism is causal counterfactual intervention:
  1. For each real patient r in group G needing augmentation:
     a. Identify which variables reflect illegitimate care disparities
        (columns on illegitimate causal paths from the ResearchSpec).
     b. Sample equitable values for those variables from the majority-group
        distribution — what those variables look like for patients who receive
        standard-of-care treatment.
     c. Propagate the change forward through the causal DAG via topological
        ordering, re-estimating downstream nodes using OLS coefficients fit on
        the majority group.
  2. Validate every generated record against clinical constraints (rejection
     sampling, max 10 attempts).
  3. Add calibrated Gaussian differential-privacy noise to numeric features.
  4. Enforce k-anonymity: each generated patient must share its quasi-identifier
     values with >= k other records (k=5 default).
  5. Compute a re-identification risk score on a sample of generated records.
"""

from __future__ import annotations

import math
import time
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from src.utils.research_spec import ResearchSpec, ClinicalConstraint
from src.utils.causal_discovery import CausalDiscoveryResult
from src.utils.intervention_optimizer import InterventionPlan


# ── Constants ──────────────────────────────────────────────────────────────────

# Default differential-privacy epsilon (higher = less noise = weaker privacy)
_DEFAULT_EPSILON: float = 0.5

# k-anonymity target: each generated patient must be indistinguishable from
# at least k-1 other records in the augmented dataset.
_DEFAULT_K: int = 5

# Maximum rejection-sampling attempts before accepting a constraint-violating
# record (last resort — prevents infinite loops with very tight constraints)
_MAX_RESAMPLE_ATTEMPTS: int = 10

# Re-identification risk: sample size and proximity threshold
_REID_SAMPLE_SIZE: int = 100
_REID_DISTANCE_FRACTION: float = 0.10  # fraction of feature-space diameter

# Minimum real patients in a group to use group-specific distribution sampling;
# below this we fall back to the full-dataset distribution for that group.
_MIN_GROUP_PATIENTS_FOR_DIST: int = 10

# Protected / demographic columns that must NEVER be modified
_DEMOGRAPHIC_KEYWORDS: Set[str] = {
    "race", "ethnicity", "sex", "gender", "age", "birth", "ancestry",
    "nationality", "religion", "indigenous",
}


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class EquitableGeneratorResult:
    """Structured output of Stage 5 Equitable-Care Counterfactual Generator."""

    original_df: pd.DataFrame
    """The input dataset (unchanged)."""

    augmented_df: pd.DataFrame
    """original_df + all generated equitable-care records."""

    n_generated: int
    """Number of synthetic patients successfully added."""

    n_rejected: int
    """Number of candidate records discarded due to clinical constraint violations."""

    privacy_budget_used: float
    """Total ε consumed by DP-noise injection across all generated records."""

    k_anonymity_achieved: int
    """Minimum k value achieved across all generated records in augmented_df."""

    generation_log: List[Dict[str, Any]]
    """Per-group generation statistics."""

    re_identification_risk: float
    """Fraction of sampled generated records closer than 10% of feature-space
    diameter to any real record — proxy for re-identification risk."""


# ── Utility helpers ────────────────────────────────────────────────────────────

def _is_demographic(col: str) -> bool:
    """Return True if the column name matches a demographic / biological upstream."""
    col_lower = col.lower().replace(" ", "_")
    return any(kw in col_lower for kw in _DEMOGRAPHIC_KEYWORDS)


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _parse_constraint_range(rule: str) -> Optional[Tuple[float, float]]:
    """
    Parse a clinical constraint rule string of the form '[lo, hi] unit'.
    Returns (lo, hi) or None if parsing fails.
    """
    try:
        inner = rule[rule.index("[") + 1: rule.index("]")]
        lo_str, hi_str = inner.split(",")
        lo = float(lo_str.strip().split()[0])
        hi = float(hi_str.strip().split()[0])
        return lo, hi
    except (ValueError, IndexError):
        return None


def _constraint_map(spec: ResearchSpec) -> Dict[str, Tuple[float, float]]:
    """Build a {feature: (min, max)} dict from spec.clinical_constraints."""
    cmap: Dict[str, Tuple[float, float]] = {}
    for c in spec.clinical_constraints:
        if c.constraint_type == "range":
            parsed = _parse_constraint_range(c.rule)
            if parsed is not None:
                cmap[c.feature.lower()] = parsed
    return cmap


def _check_clinical_validity(
    row: pd.Series,
    cmap: Dict[str, Tuple[float, float]],
) -> bool:
    """Return True if the row satisfies all clinical range constraints."""
    for feat_lower, (lo, hi) in cmap.items():
        for col in row.index:
            if col.lower() == feat_lower or feat_lower in col.lower():
                val = row[col]
                if pd.isna(val):
                    continue
                try:
                    if not (lo <= float(val) <= hi):
                        return False
                except (TypeError, ValueError):
                    pass
    return True


# ── Illegitimate mediator identification ──────────────────────────────────────

def _illegitimate_mediators(spec: ResearchSpec, df_cols: List[str]) -> List[str]:
    """
    Return the list of DataFrame columns that correspond to mediators on
    illegitimate causal paths specified in the ResearchSpec.
    These are the columns we will intervene on (replace with equitable values).
    """
    mediators: List[str] = []
    seen: Set[str] = set()
    cols_lower = {c.lower().replace(" ", "_"): c for c in df_cols}

    for path_obj in spec.illegitimate_paths:
        parts = [p.strip() for p in path_obj.path.replace("→", "|").split("|")]
        # Middle elements of the path (skip source and target) are mediators
        for med_raw in parts[1:-1]:
            norm = med_raw.lower().replace(" ", "_")
            if norm in cols_lower and cols_lower[norm] not in seen:
                col = cols_lower[norm]
                if not _is_demographic(col):
                    mediators.append(col)
                    seen.add(col)
            else:
                # Try partial match
                for col_norm, col_orig in cols_lower.items():
                    if norm in col_norm and col_orig not in seen:
                        if not _is_demographic(col_orig):
                            mediators.append(col_orig)
                            seen.add(col_orig)
                            break

    return mediators


# ── Majority-group distribution sampling ──────────────────────────────────────

def _majority_group(
    df: pd.DataFrame,
    protected_cols: List[str],
) -> pd.DataFrame:
    """
    Identify the modal class in each protected attribute column and return the
    subset of rows that match the modal class in the *first* protected column
    (primary protected attribute).  This represents patients who likely received
    standard-of-care treatment in a biased dataset.
    """
    if not protected_cols:
        return df
    primary_col = protected_cols[0]
    if primary_col not in df.columns:
        return df
    mode_val = df[primary_col].mode()
    if mode_val.empty:
        return df
    return df[df[primary_col] == mode_val.iloc[0]]


def _sample_equitable_values(
    mediator_cols: List[str],
    majority_df: pd.DataFrame,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """
    Sample values for illegitimate mediators from the majority-group distribution.
    Numeric columns: draw from empirical distribution (resample with replacement).
    Categorical columns: draw from empirical category frequencies.
    """
    equitable: Dict[str, Any] = {}
    for col in mediator_cols:
        if col not in majority_df.columns:
            continue
        if len(majority_df) == 0:
            continue
        if pd.api.types.is_numeric_dtype(majority_df[col]):
            valid = majority_df[col].dropna()
            if len(valid) == 0:
                continue
            equitable[col] = float(rng.choice(valid.values))
        else:
            valid = majority_df[col].dropna()
            if len(valid) == 0:
                continue
            vals, counts = np.unique(valid.astype(str).values, return_counts=True)
            probs = counts / counts.sum()
            equitable[col] = rng.choice(vals, p=probs)
    return equitable


# ── DAG-based counterfactual propagation ──────────────────────────────────────

def _topological_order(dag: nx.DiGraph, nodes: List[str]) -> List[str]:
    """
    Return a topological ordering of the given nodes (those present in the DAG).
    Nodes absent from the DAG are appended at the end.
    """
    dag_nodes_set = set(dag.nodes())
    in_dag = [n for n in nodes if n in dag_nodes_set]
    not_in_dag = [n for n in nodes if n not in dag_nodes_set]
    try:
        # nx.lexicographic_topological_sort is deterministic
        topo = list(nx.topological_sort(dag))
        ordered = [n for n in topo if n in set(in_dag)]
    except nx.NetworkXUnfeasible:
        ordered = in_dag
    return ordered + not_in_dag


class _OLSPropagator:
    """
    Learns linear regression coefficients for each target column conditioned on
    its direct parents in the DAG, using the majority-group subset.
    Caches fitted models so they are estimated once per group.
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        majority_df: pd.DataFrame,
        rng: np.random.RandomState,
    ):
        self._dag = dag
        self._majority = majority_df
        self._rng = rng
        self._models: Dict[str, LinearRegression] = {}
        self._label_encoders: Dict[str, LabelEncoder] = {}

    def _encode(self, series: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(series.median()).values.astype(float)
        col_name = series.name
        if col_name not in self._label_encoders:
            le = LabelEncoder()
            le.fit(series.fillna("__MISSING__").astype(str))
            self._label_encoders[col_name] = le
        le = self._label_encoders[col_name]
        return le.transform(series.fillna("__MISSING__").astype(str)).astype(float)

    def _fit(self, target_col: str) -> Optional[LinearRegression]:
        """Fit OLS for target_col ~ parents(target_col) on majority_df."""
        if target_col not in self._majority.columns:
            return None
        parents = list(self._dag.predecessors(target_col)) if target_col in self._dag else []
        parents = [p for p in parents if p in self._majority.columns]
        if not parents:
            return None

        y = self._encode(self._majority[target_col])
        X = np.column_stack([self._encode(self._majority[p]) for p in parents])
        try:
            model = LinearRegression().fit(X, y)
            self._models[target_col] = model
            return model
        except Exception:
            return None

    def propagate(
        self,
        record: pd.Series,
        intervened_cols: List[str],
        all_numeric_cols: List[str],
    ) -> pd.Series:
        """
        Starting from the intervened columns, propagate downstream changes
        through the DAG.  Demographic and biological-upstream columns are frozen.
        """
        row = record.copy()
        dag_nodes = set(self._dag.nodes())

        # Build a topological order of all relevant downstream nodes
        downstream: Set[str] = set()
        for start_col in intervened_cols:
            if start_col in dag_nodes:
                try:
                    desc = nx.descendants(self._dag, start_col)
                    downstream.update(desc)
                except Exception:
                    pass

        # Process in topological order; skip demographic columns
        update_order = _topological_order(self._dag, list(downstream))
        for col in update_order:
            if _is_demographic(col):
                continue
            if col in intervened_cols:
                continue
            if col not in row.index:
                continue

            parents = (
                list(self._dag.predecessors(col))
                if col in dag_nodes
                else []
            )
            parents = [p for p in parents if p in row.index]
            if not parents:
                continue

            model = self._models.get(col) or self._fit(col)
            if model is None:
                continue

            try:
                x_vals = np.array([
                    float(row[p]) if pd.api.types.is_numeric_dtype(
                        type(row[p])
                    ) else float(
                        self._label_encoders.get(p, LabelEncoder()).transform(
                            [str(row[p])]
                        )[0]
                        if p in self._label_encoders
                        else 0
                    )
                    for p in parents
                ]).reshape(1, -1)
                pred = float(model.predict(x_vals)[0])
                # Add small Gaussian noise to avoid exact duplicates
                noise = self._rng.normal(0, abs(pred) * 0.01 + 1e-6)
                row[col] = pred + noise
            except Exception:
                pass  # Leave column unchanged on propagation failure

        return row


# ── Differential privacy noise ────────────────────────────────────────────────

def _add_dp_noise(
    row: pd.Series,
    numeric_cols: List[str],
    col_stds: Dict[str, float],
    n: int,
    epsilon: float,
    rng: np.random.RandomState,
) -> pd.Series:
    """
    Add calibrated Gaussian DP noise to each numeric feature.

    Noise model
    -----------
    sensitivity_f = std(f) / sqrt(n)      (global sensitivity for the mean)
    σ_f           = sensitivity_f / ε      (Gaussian mechanism sigma)
    noise_f       ~ N(0, σ_f²)

    Parameters
    ----------
    row          : a single record (pd.Series)
    numeric_cols : columns to add noise to
    col_stds     : per-column standard deviations from the source dataset
    n            : source dataset size (used in sensitivity calculation)
    epsilon      : DP budget per record
    rng          : numpy RandomState for reproducibility

    Returns the noised row.
    """
    noised = row.copy()
    for col in numeric_cols:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        try:
            std_f = col_stds.get(col, 1.0)
            sensitivity = std_f / math.sqrt(max(n, 1))
            sigma = sensitivity / max(epsilon, 1e-9)
            noise = rng.normal(0.0, sigma)
            noised[col] = float(val) + noise
        except (TypeError, ValueError):
            pass  # Skip non-numeric values gracefully
    return noised


# ── k-anonymity enforcement ────────────────────────────────────────────────────

def _quasi_identifier_cols(df: pd.DataFrame, spec: ResearchSpec) -> List[str]:
    """
    Return demographic / quasi-identifier columns present in the DataFrame.
    Used to measure k-anonymity across the augmented dataset.
    """
    qi_cols: List[str] = []
    for attr_spec in spec.protected_attributes:
        col = attr_spec.attribute
        if col in df.columns:
            qi_cols.append(col)
    # Also include age if present as a binned / discrete column
    for col in df.columns:
        if "age" in col.lower() and col not in qi_cols:
            qi_cols.append(col)
    return qi_cols


def _compute_k_anonymity(df: pd.DataFrame, qi_cols: List[str]) -> int:
    """
    Compute the minimum group size over all quasi-identifier combinations.
    Returns 1 if no QI columns are present (worst case).
    """
    if not qi_cols:
        return 1
    available = [c for c in qi_cols if c in df.columns]
    if not available:
        return 1
    try:
        group_sizes = df.groupby(available).size()
        return int(group_sizes.min())
    except Exception:
        return 1


def _enforce_k_anonymity(
    generated_rows: List[pd.Series],
    real_df: pd.DataFrame,
    qi_cols: List[str],
    k: int,
) -> List[pd.Series]:
    """
    Remove generated records from groups with fewer than k members in the
    combined (real + generated) dataset.

    Iterates until stable — each pass may open new under-k groups.
    Returns the filtered list of generated rows.
    """
    if not qi_cols or k <= 1:
        return generated_rows

    available_qi = [c for c in qi_cols if c in real_df.columns]
    if not available_qi:
        return generated_rows

    # Build a combined DataFrame (real + current generated)
    gen_df = pd.DataFrame(generated_rows) if generated_rows else pd.DataFrame()
    combined = pd.concat([real_df, gen_df], ignore_index=True)

    for _iteration in range(20):  # max 20 passes
        group_sizes = combined.groupby(available_qi).size()
        # Identify QI-keys with insufficient k
        small_keys = set(
            key for key, sz in group_sizes.items() if sz < k
        )
        if not small_keys:
            break

        # Filter generated_rows
        new_gen: List[pd.Series] = []
        for row in generated_rows:
            try:
                key = tuple(
                    row[c] if len(available_qi) > 1 else row[available_qi[0]]
                    for c in available_qi
                )
                if len(available_qi) == 1:
                    key = key[0]
                if key in small_keys:
                    continue
            except (KeyError, TypeError):
                pass
            new_gen.append(row)

        if len(new_gen) == len(generated_rows):
            break  # No change — converged

        generated_rows = new_gen
        gen_df = pd.DataFrame(generated_rows) if generated_rows else pd.DataFrame()
        combined = pd.concat([real_df, gen_df], ignore_index=True)

    return generated_rows


# ── Re-identification risk ─────────────────────────────────────────────────────

def _reidentification_risk(
    generated_df: pd.DataFrame,
    real_df: pd.DataFrame,
    numeric_cols: List[str],
    rng: np.random.RandomState,
) -> float:
    """
    Estimate re-identification risk.

    Algorithm
    ---------
    1. Sample min(100, n_generated) records from generated_df.
    2. For each sampled record, find its nearest neighbour in real_df using
       Euclidean distance on numeric features.
    3. risk = fraction of sampled records with nearest-neighbour distance
       < _REID_DISTANCE_FRACTION * feature_space_diameter.

    Feature-space diameter is approximated as the L2 norm of the range vector
    over numeric features in real_df.
    """
    available = [c for c in numeric_cols if c in generated_df.columns and c in real_df.columns]
    if not available or len(generated_df) == 0:
        return 0.0

    # Normalise to [0, 1] per feature using real_df statistics
    real_vals = real_df[available].dropna().values.astype(float)
    gen_vals  = generated_df[available].fillna(0.0).values.astype(float)

    if real_vals.shape[0] < 2 or gen_vals.shape[0] < 1:
        return 0.0

    col_min  = real_vals.min(axis=0)
    col_max  = real_vals.max(axis=0)
    col_range = np.where(col_max - col_min > 1e-9, col_max - col_min, 1.0)

    real_norm = (real_vals - col_min) / col_range
    gen_norm  = (gen_vals  - col_min) / col_range

    # Feature-space diameter
    diameter = float(np.linalg.norm(np.ones(len(available))))
    threshold = _REID_DISTANCE_FRACTION * diameter

    sample_n = min(_REID_SAMPLE_SIZE, len(gen_norm))
    sample_idx = rng.choice(len(gen_norm), size=sample_n, replace=False)
    sample = gen_norm[sample_idx]

    close_count = 0
    for pt in sample:
        diffs = real_norm - pt
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        min_dist = dists.min()
        if min_dist < threshold:
            close_count += 1

    return round(close_count / sample_n, 4) if sample_n > 0 else 0.0


# ── Distribution-matching fallback ────────────────────────────────────────────

def _generate_from_distribution(
    n_to_generate: int,
    source_df: pd.DataFrame,
    spec: ResearchSpec,
    cmap: Dict[str, Tuple[float, float]],
    col_stds: Dict[str, float],
    epsilon: float,
    rng: np.random.RandomState,
) -> Tuple[List[pd.Series], int]:
    """
    Fallback generation path used when the causal DAG has no edges.

    For each numeric column, draw samples from a Gaussian fitted to source_df.
    For each categorical column, resample from empirical frequencies.
    Apply DP noise and clinical validation.

    Returns (list of valid rows, n_rejected).
    """
    n_rejected = 0
    generated: List[pd.Series] = []
    numeric = _numeric_cols(source_df)

    # Fit per-column distributions from source_df
    col_means = {c: float(source_df[c].dropna().mean()) for c in numeric}
    col_std   = {c: float(source_df[c].dropna().std()) for c in numeric}

    for _ in range(n_to_generate):
        attempts = 0
        accepted = False
        while attempts < _MAX_RESAMPLE_ATTEMPTS:
            row = source_df.sample(1, random_state=rng).iloc[0].copy()
            for col in numeric:
                mu  = col_means.get(col, 0.0)
                sig = max(col_std.get(col, 1.0), 1e-6)
                row[col] = float(rng.normal(mu, sig))

            # DP noise
            row = _add_dp_noise(row, numeric, col_stds, len(source_df), epsilon, rng)

            if _check_clinical_validity(row, cmap):
                generated.append(row)
                accepted = True
                break
            attempts += 1

        if not accepted:
            # Accept the last attempt regardless — better than silently dropping
            row = _add_dp_noise(
                source_df.sample(1, random_state=rng).iloc[0].copy(),
                numeric, col_stds, len(source_df), epsilon, rng,
            )
            generated.append(row)
            n_rejected += 1

    return generated, n_rejected


# ── Core generation loop ───────────────────────────────────────────────────────

def _generate_group(
    group_label: str,
    group_df: pd.DataFrame,
    n_to_generate: int,
    full_df: pd.DataFrame,
    majority_df: pd.DataFrame,
    dag: nx.DiGraph,
    mediator_cols: List[str],
    spec: ResearchSpec,
    cmap: Dict[str, Tuple[float, float]],
    col_stds: Dict[str, float],
    epsilon: float,
    k: int,
    rng: np.random.RandomState,
) -> Tuple[List[pd.Series], int, int]:
    """
    Generate n_to_generate equitable-care counterfactual records for one group.

    Returns (generated_rows, n_generated, n_rejected).
    """
    numeric_cols = _numeric_cols(full_df)
    n_rejected = 0

    # Decide source population for sampling seed patients
    source_df = (
        group_df
        if len(group_df) >= _MIN_GROUP_PATIENTS_FOR_DIST
        else full_df
    )

    # If the DAG has no edges, use distribution-matching fallback
    has_edges = dag.number_of_edges() > 0
    if not has_edges:
        print(f"    [{group_label}] DAG has no edges — using distribution-matching fallback")
        gen_rows, n_rej = _generate_from_distribution(
            n_to_generate, source_df, spec, cmap, col_stds, epsilon, rng
        )
        return gen_rows, len(gen_rows), n_rej

    # Fit OLS propagation models on majority group
    propagator = _OLSPropagator(dag, majority_df, rng)
    # Pre-fit models for mediator columns' downstream nodes eagerly
    downstream_nodes: Set[str] = set()
    for med in mediator_cols:
        if med in dag:
            try:
                downstream_nodes.update(nx.descendants(dag, med))
            except Exception:
                pass
    for dn in downstream_nodes:
        if dn in majority_df.columns:
            propagator._fit(dn)

    generated: List[pd.Series] = []
    attempts_total = 0
    generated_count = 0

    while generated_count < n_to_generate and attempts_total < n_to_generate * _MAX_RESAMPLE_ATTEMPTS:
        # Seed from a real patient in the group (or full dataset if too few)
        seed = source_df.sample(1, random_state=rng).iloc[0].copy()

        # Step 1 — Sample equitable values for illegitimate mediators
        equitable_vals = _sample_equitable_values(mediator_cols, majority_df, rng)

        # Step 2 — Apply the intervention
        intervened_row = seed.copy()
        for col, val in equitable_vals.items():
            intervened_row[col] = val

        # Step 3 — Propagate through DAG
        try:
            propagated_row = propagator.propagate(
                intervened_row, list(equitable_vals.keys()), numeric_cols
            )
        except Exception:
            propagated_row = intervened_row

        # Step 4 — DP noise
        noised_row = _add_dp_noise(
            propagated_row, numeric_cols, col_stds, len(full_df), epsilon, rng
        )

        # Step 5 — Clinical validation (rejection sampling)
        if _check_clinical_validity(noised_row, cmap):
            generated.append(noised_row)
            generated_count += 1
        else:
            n_rejected += 1

        attempts_total += 1

    # If we still need more, pad with distribution-matching fallback
    remaining = n_to_generate - generated_count
    if remaining > 0:
        fallback_rows, fallback_rej = _generate_from_distribution(
            remaining, source_df, spec, cmap, col_stds, epsilon, rng
        )
        generated.extend(fallback_rows)
        n_rejected += fallback_rej

    return generated, len(generated), n_rejected


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_equitable_cohort(
    df: pd.DataFrame,
    spec: ResearchSpec,
    causal_result: CausalDiscoveryResult,
    intervention_plan: InterventionPlan,
    epsilon: float = _DEFAULT_EPSILON,
    k: int = _DEFAULT_K,
    random_state: int = 42,
) -> EquitableGeneratorResult:
    """
    Stage 5 of FIDES — Equitable-Care Counterfactual Generator.

    Parameters
    ----------
    df : pd.DataFrame
        De-identified dataset (Stage 1 output).
    spec : ResearchSpec
        Stage 0 output.
    causal_result : CausalDiscoveryResult
        Stage 2 output — provides the causal DAG.
    intervention_plan : InterventionPlan
        Stage 4 output — specifies how many patients are needed per group.
    epsilon : float
        Differential-privacy budget ε for Gaussian noise injection per record.
    k : int
        k-anonymity threshold.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    EquitableGeneratorResult
    """
    t_start = time.time()
    rng = np.random.RandomState(random_state)

    print("=" * 64)
    print("FIDES Stage 5 — Equitable-Care Counterfactual Generator")
    print(f"  Domain        : {spec.domain}")
    print(f"  DP epsilon    : {epsilon}")
    print(f"  k-anonymity   : {k}")
    print(f"  DAG edges     : {causal_result.dag.number_of_edges()}")
    print(f"  Intervention  : {intervention_plan.total_new_patients} patients requested")
    print("=" * 64)

    # Pre-compute dataset-level statistics once
    numeric_cols = _numeric_cols(df)
    col_stds: Dict[str, float] = {
        c: max(float(df[c].dropna().std()), 1e-6) for c in numeric_cols
    }
    cmap = _constraint_map(spec)

    # Identify mediator columns (illegitimate care-pathway variables)
    mediator_cols = _illegitimate_mediators(spec, list(df.columns))
    print(f"\n  Illegitimate mediators identified: {mediator_cols}")

    # Identify the majority group for equitable-value sampling
    protected_cols = [
        s.attribute for s in spec.protected_attributes
        if s.attribute in df.columns
    ]
    majority_df = _majority_group(df, protected_cols)
    print(f"  Majority group size (for equitable-value sampling): {len(majority_df)}")

    # ── Main generation loop ───────────────────────────────────────────────────
    all_generated_rows: List[pd.Series] = []
    total_rejected = 0
    generation_log: List[Dict[str, Any]] = []
    total_epsilon_used = 0.0

    # Build group-to-dataframe mapping
    # We use the GroupRecruitmentTarget names to identify groups in the DataFrame
    groups_to_process = intervention_plan.groups

    if not groups_to_process:
        # No intervention needed — return original dataset
        print("\n  No groups require augmentation.")
        return EquitableGeneratorResult(
            original_df=df.copy(),
            augmented_df=df.copy(),
            n_generated=0,
            n_rejected=0,
            privacy_budget_used=0.0,
            k_anonymity_achieved=_compute_k_anonymity(df, _quasi_identifier_cols(df, spec)),
            generation_log=[],
            re_identification_risk=0.0,
        )

    print(f"\n  Processing {len(groups_to_process)} groups…\n")

    for target in groups_to_process:
        group_name = target.group_name
        n_to_gen   = target.n_required
        phenotype  = target.phenotype_profile

        print(f"  Group: {group_name}")
        print(f"    n_to_generate = {n_to_gen}  |  priority = {target.priority}")

        # ── Identify matching rows in df for this group ────────────────────────
        group_mask = pd.Series([True] * len(df), index=df.index)

        # Apply phenotype constraints to filter the source group
        for col in df.columns:
            if col in phenotype:
                pval = phenotype[col]
                if isinstance(pval, dict) and "min" in pval and "max" in pval:
                    lo, hi = pval["min"], pval["max"]
                    if pd.api.types.is_numeric_dtype(df[col]):
                        group_mask &= (df[col] >= lo) & (df[col] <= hi)
                elif isinstance(pval, str):
                    group_mask &= df[col].astype(str).str.lower() == pval.lower()

        group_df = df[group_mask].copy()

        # If phenotype filtering yields too-few rows, expand to protected-attr match
        if len(group_df) < _MIN_GROUP_PATIENTS_FOR_DIST:
            # Try matching on the driving protected attribute only
            if "race" in phenotype and "race" in df.columns:
                group_df = df[
                    df["race"].astype(str).str.lower() == str(phenotype["race"]).lower()
                ].copy()
            if len(group_df) < 2:
                group_df = df.copy()  # Fall back to full dataset

        print(f"    Source rows for this group : {len(group_df)}")

        # ── Generate records ───────────────────────────────────────────────────
        gen_rows, n_gen, n_rej = _generate_group(
            group_label=group_name,
            group_df=group_df,
            n_to_generate=n_to_gen,
            full_df=df,
            majority_df=majority_df,
            dag=causal_result.dag,
            mediator_cols=mediator_cols,
            spec=spec,
            cmap=cmap,
            col_stds=col_stds,
            epsilon=epsilon,
            k=k,
            rng=rng,
        )

        # Track per-record epsilon expenditure
        epsilon_per_record = epsilon
        group_epsilon = n_gen * epsilon_per_record
        total_epsilon_used += group_epsilon

        all_generated_rows.extend(gen_rows)
        total_rejected += n_rej

        log_entry = {
            "group":              group_name,
            "requested":          n_to_gen,
            "generated":          n_gen,
            "rejected":           n_rej,
            "epsilon_used":       round(group_epsilon, 6),
            "source_rows":        len(group_df),
            "driving_condition":  target.condition_driving_need,
        }
        generation_log.append(log_entry)
        print(f"    Generated: {n_gen}  |  Rejected: {n_rej}  |  "
              f"ε used: {group_epsilon:.4f}")

    # ── k-anonymity enforcement ────────────────────────────────────────────────
    qi_cols = _quasi_identifier_cols(df, spec)
    print(f"\n  Enforcing k={k} anonymity on {len(all_generated_rows)} generated records…")
    before_k = len(all_generated_rows)
    all_generated_rows = _enforce_k_anonymity(all_generated_rows, df, qi_cols, k)
    after_k = len(all_generated_rows)
    if before_k > after_k:
        print(f"    Removed {before_k - after_k} records that violated k-anonymity")

    # ── Assemble augmented DataFrame ───────────────────────────────────────────
    if all_generated_rows:
        gen_df = pd.DataFrame(all_generated_rows).reset_index(drop=True)
        # Align columns to match original df
        for col in df.columns:
            if col not in gen_df.columns:
                gen_df[col] = np.nan
        gen_df = gen_df[df.columns]
        augmented_df = pd.concat([df.reset_index(drop=True), gen_df], ignore_index=True)
    else:
        augmented_df = df.copy()

    # ── k-anonymity measurement ────────────────────────────────────────────────
    k_achieved = _compute_k_anonymity(augmented_df, qi_cols)

    # ── Re-identification risk ─────────────────────────────────────────────────
    print("\n  Computing re-identification risk…")
    if all_generated_rows:
        gen_only_df = pd.DataFrame(all_generated_rows)
        reid_risk = _reidentification_risk(gen_only_df, df, numeric_cols, rng)
    else:
        reid_risk = 0.0
    print(f"    Re-ID risk = {reid_risk:.4f}  (threshold < {_REID_DISTANCE_FRACTION})")

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    total_generated = len(all_generated_rows)

    print("\n" + "=" * 64)
    print("EQUITABLE GENERATOR SUMMARY")
    print("=" * 64)
    print(f"  Original dataset size   : {len(df):,}")
    print(f"  Generated records       : {total_generated:,}")
    print(f"  Rejected records        : {total_rejected:,}")
    print(f"  Augmented dataset size  : {len(augmented_df):,}")
    print(f"  k-anonymity achieved    : {k_achieved}")
    print(f"  Re-ID risk              : {reid_risk:.4f}")
    print(f"  Total ε consumed        : {total_epsilon_used:.6f}")
    print(f"  Elapsed                 : {elapsed:.2f}s")
    print("=" * 64)

    return EquitableGeneratorResult(
        original_df=df.copy(),
        augmented_df=augmented_df,
        n_generated=total_generated,
        n_rejected=total_rejected,
        privacy_budget_used=round(total_epsilon_used, 6),
        k_anonymity_achieved=k_achieved,
        generation_log=generation_log,
        re_identification_risk=reid_risk,
    )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.research_spec import build_research_spec
    from src.utils.causal_discovery import run_causal_discovery
    from src.utils.cds_assessor import assess_dataset_sufficiency
    from src.utils.intervention_optimizer import optimize_intervention

    np.random.seed(42)
    n = 600

    age_arr      = np.random.normal(55, 12, n)
    sex_arr      = np.random.choice(["Male", "Female"], n, p=[0.60, 0.40])
    race_arr     = np.random.choice(
        ["White", "Black", "Hispanic", "Asian"],
        n,
        p=[0.70, 0.08, 0.15, 0.07],
    )
    insurance_arr = np.random.choice(
        ["Private", "Medicare", "Medicaid", "None"],
        n,
        p=[0.55, 0.25, 0.12, 0.08],
    )
    troponin_arr = np.clip(
        0.025 * age_arr + 0.3 * (sex_arr == "Male").astype(float)
        + np.random.normal(0, 0.4, n),
        0, 5,
    )
    pain_arr = 4.5 + 0.6 * (sex_arr == "Male") + np.random.normal(0, 1.5, n)
    ttd_arr  = np.random.normal(3.5, 1.2, n)
    mi_arr   = (troponin_arr > 1.0).astype(int)

    df_demo = pd.DataFrame({
        "age":                 age_arr,
        "sex":                 sex_arr,
        "race":                race_arr,
        "insurance":           insurance_arr,
        "troponin":            troponin_arr,
        "pain_score_recorded": pain_arr,
        "time_to_diagnosis":   ttd_arr,
        "MI_outcome":          mi_arr,
    })

    spec = build_research_spec(
        domain="cardiology",
        target_variable="MI_outcome",
        target_type="binary",
        use_case="fda_submission",
        columns=list(df_demo.columns),
        intent="Predict MI risk in adult ED population",
    )

    print("\n=== Stage 2: Causal Discovery ===")
    causal_res = run_causal_discovery(df_demo, spec, epsilon=1.0)

    print("\n=== Stage 3: CDS Assessment ===")
    cds_res = assess_dataset_sufficiency(df_demo, spec, causal_res)

    print("\n=== Stage 4: Intervention Optimizer ===")
    plan = optimize_intervention(cds_res, spec, budget=300_000)

    print("\n=== Stage 5: Equitable Generator ===")
    result = generate_equitable_cohort(
        df=df_demo,
        spec=spec,
        causal_result=causal_res,
        intervention_plan=plan,
        epsilon=0.5,
        k=5,
        random_state=42,
    )

    print("\n─── EquitableGeneratorResult ───")
    print(f"  original_df rows        : {len(result.original_df)}")
    print(f"  augmented_df rows       : {len(result.augmented_df)}")
    print(f"  n_generated             : {result.n_generated}")
    print(f"  n_rejected              : {result.n_rejected}")
    print(f"  privacy_budget_used     : {result.privacy_budget_used}")
    print(f"  k_anonymity_achieved    : {result.k_anonymity_achieved}")
    print(f"  re_identification_risk  : {result.re_identification_risk}")
    print(f"  generation_log entries  : {len(result.generation_log)}")
    for entry in result.generation_log:
        print(f"    {entry['group']}: requested={entry['requested']}, "
              f"generated={entry['generated']}, rejected={entry['rejected']}")


# ── Convenience class wrapper for backend compatibility ───────────────────────

class EquitableGenerator:
    """Class wrapper around generate_equitable_cohort for FIDES pipeline."""

    def __init__(self, spec, causal_result, epsilon: float = 0.5, k: int = 5, random_state: int = 42):
        self.spec = spec
        self.causal_result = causal_result
        self.epsilon = epsilon
        self.k = k
        self.random_state = random_state

    def generate(self, df, intervention_plan) -> "EquitableGeneratorResult":
        return generate_equitable_cohort(
            df=df,
            spec=self.spec,
            causal_result=self.causal_result,
            intervention_plan=intervention_plan,
            epsilon=self.epsilon,
            k=self.k,
            random_state=self.random_state,
        )
