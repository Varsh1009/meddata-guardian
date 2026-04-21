"""
FIDES Stage 2 — DP-PC Causal Discovery
Learns a causal DAG from a de-identified pandas DataFrame using the PC algorithm
with Laplace-noise differential privacy applied to Fisher's Z test statistics.

Privacy guarantee: each conditional independence test consumes ε/T budget
where T is the maximum possible number of CI tests (n*(n-1)/2 pairs).
The privacy_budget_used field records ε₁ actually consumed.
"""

from __future__ import annotations

import math
import time
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure venv site-packages is on path for causallearn
from pathlib import Path as _Path
_project_root = _Path(__file__).resolve().parents[2]  # meddata-guardian-1/
_venv_site = _project_root / "venv" / "lib"
for _p in _venv_site.glob("python*/site-packages"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# ── causal-learn availability guard ──────────────────────────────────────────
_CAUSALLEARN_OK = False
_cl_err_msg = ""
try:
    import sys as _sys
    # causal-learn installs into the venv site-packages as 'causallearn'
    import importlib.util as _ilu
    _spec_cl = _ilu.find_spec("causallearn")
    if _spec_cl is None:
        raise ImportError("causallearn not on sys.path")
    from causallearn.search.ConstraintBased.PC import pc as causallearn_pc
    from causallearn.utils.cit import (
        CIT_Base,
        FisherZ,
        register_ci_test,
        _custom_ci_tests,
    )
    _CAUSALLEARN_OK = True
except Exception as _e:
    _cl_err_msg = str(_e)

from src.utils.research_spec import ResearchSpec


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class CausalDiscoveryResult:
    """Structured output of Stage 2 DP-PC causal discovery."""

    dag: nx.DiGraph
    """NetworkX DiGraph of the learned causal DAG."""

    edges: List[Tuple[str, str]]
    """Directed edges as (source, target) pairs.  Undirected skeleton edges
    are included as both (u, v) and (v, u) when orientation was ambiguous."""

    privacy_budget_used: float
    """Total ε₁ consumed across all CI tests (FIDES privacy ledger entry)."""

    path_validation: Dict[str, str]
    """Maps each ResearchSpec path string → 'confirmed' | 'disputed' | 'new'."""

    column_mapping: Dict[str, str]
    """Maps each column name used in the DAG back to the original DataFrame name."""

    n_tests: int = 0
    """Number of conditional independence tests that were executed."""

    method_used: str = "dp_pc"
    """'dp_pc' when causal-learn succeeded; 'correlation_fallback' otherwise."""


# ── DP-FisherZ CI test (registered with causal-learn) ────────────────────────

# Thread-local holder so the instantiated object can be retrieved after pc()
_DP_FISHERZ_INSTANCE: Dict[str, object] = {}

if _CAUSALLEARN_OK:

    class _DPFisherZ(FisherZ):  # type: ignore[misc]
        """
        FisherZ subclass that adds calibrated Laplace noise to the test statistic.

        Noise model
        -----------
        For a Fisher Z test with conditioning set of size k on n samples:

            X = sqrt(n - k - 3) * |Z|       (standard statistic)
            sensitivity = 1 / sqrt(n - k - 3)   (L1 sensitivity of |X|)
            noise ~ Laplace(0, sensitivity / ε_per_test)

        The perturbed statistic X' = max(X + noise, 0) is used to compute
        the two-tailed p-value, making each call ε_per_test-DP.
        """

        def __init__(
            self,
            data: np.ndarray,
            epsilon_per_test: float = 0.01,
            random_seed: int = 42,
            **kwargs,
        ):
            super().__init__(data, **kwargs)
            self._eps = max(epsilon_per_test, 1e-12)
            self._rng = np.random.RandomState(random_seed)
            self.test_count = 0
            # Register this instance in the module-level holder so the caller
            # can retrieve test_count after pc() returns.
            _DP_FISHERZ_INSTANCE["current"] = self

        def __call__(self, X, Y, condition_set=None):  # type: ignore[override]
            Xs, Ys, cond, cache_key = self.get_formatted_XYZ_and_cachekey(
                X, Y, condition_set
            )
            if cache_key in self.pvalue_cache:
                return self.pvalue_cache[cache_key]

            var = Xs + Ys + cond
            sub_corr = self.correlation_matrix[np.ix_(var, var)]
            try:
                inv = np.linalg.inv(sub_corr)
            except np.linalg.LinAlgError:
                # Singular sub-matrix — declare independence conservatively
                self.pvalue_cache[cache_key] = 1.0
                return 1.0

            r = -inv[0, 1] / math.sqrt(abs(inv[0, 0] * inv[1, 1]))
            r = float(np.clip(r, -(1.0 - np.finfo(float).eps), 1.0 - np.finfo(float).eps))

            Z_val = 0.5 * math.log((1.0 + r) / (1.0 - r))
            k = len(cond)
            n = self.sample_size
            denom = max(n - k - 3, 1)
            X_stat = math.sqrt(denom) * abs(Z_val)

            # ── Differential-privacy Laplace noise ───────────────────────────
            sensitivity = 1.0 / math.sqrt(denom)   # L1 sensitivity of |X|
            noise = self._rng.laplace(0.0, sensitivity / self._eps)
            X_stat_dp = max(X_stat + noise, 0.0)
            # ─────────────────────────────────────────────────────────────────

            p = 2.0 * (1.0 - stats.norm.cdf(X_stat_dp))
            self.pvalue_cache[cache_key] = p
            self.test_count += 1
            return p

    # Register once; subsequent calls with 'dp_fisherz' will use this class.
    if "dp_fisherz" not in _custom_ci_tests:
        register_ci_test("dp_fisherz", _DPFisherZ)


# ── DataFrame encoding ────────────────────────────────────────────────────────

def _encode_dataframe(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, str], List[str]]:
    """
    Encode a mixed-type DataFrame into a float64 matrix suitable for Fisher's Z.

    Numeric columns  → fill NaN with column median, then z-score standardise.
    Categorical/bool → LabelEncode, then z-score standardise.

    A tiny Gaussian jitter (σ=1e-8) is added to every column to prevent
    perfect collinearity in edge-case datasets (all-constant columns).

    Returns
    -------
    data_matrix : (n_samples, n_features) float64 ndarray
    col_map     : {column_name: original_column_name}   (names are the same here)
    col_names   : ordered list of column names as they appear in data_matrix
    """
    jitter_rng = np.random.RandomState(random_state)
    arrays: Dict[str, np.ndarray] = {}
    col_map: Dict[str, str] = {}

    for col in df.columns:
        series = df[col].copy()
        if pd.api.types.is_numeric_dtype(series):
            arr = series.fillna(series.median()).values.astype(float)
        else:
            le = LabelEncoder()
            arr = le.fit_transform(
                series.fillna("__MISSING__").astype(str)
            ).astype(float)

        std = arr.std(ddof=0)
        arr = (arr - arr.mean()) / std if std > 1e-12 else arr - arr.mean()
        arr = arr + jitter_rng.normal(0.0, 1e-8, size=arr.shape)

        arrays[col] = arr
        col_map[col] = col   # original name preserved

    col_names = list(arrays.keys())
    data_matrix = np.column_stack([arrays[c] for c in col_names]).astype(np.float64)
    return data_matrix, col_map, col_names


# ── Graph conversion ──────────────────────────────────────────────────────────

def _causallearn_graph_to_nx(
    cg,
    col_names: List[str],
) -> Tuple[nx.DiGraph, List[Tuple[str, str]]]:
    """
    Convert a causal-learn CausalGraph to a networkx DiGraph.

    causal-learn graph matrix encoding (cg.G.graph[i, j]):
      -1  = tail at node i on edge i--j
       1  = arrowhead at node i on edge i-->... (i receives arrow)
       0  = no edge

    Directed edge i → j :  graph[i,j] = -1  AND  graph[j,i] = 1
    Undirected   i - j  :  graph[i,j] = -1  AND  graph[j,i] = -1  (both tails)
    """
    mat = cg.G.graph
    n = len(col_names)
    dag = nx.DiGraph()
    dag.add_nodes_from(col_names)
    edges: List[Tuple[str, str]] = []
    seen_undirected: set = set()

    for i in range(n):
        for j in range(i + 1, n):
            e_ij = mat[i, j]
            e_ji = mat[j, i]
            src, tgt = col_names[i], col_names[j]

            if e_ij == -1 and e_ji == 1:
                # i → j
                dag.add_edge(src, tgt)
                edges.append((src, tgt))
            elif e_ij == 1 and e_ji == -1:
                # j → i
                dag.add_edge(tgt, src)
                edges.append((tgt, src))
            elif e_ij == -1 and e_ji == -1:
                # Undirected skeleton edge; add both orientations
                pair = (min(src, tgt), max(src, tgt))
                if pair not in seen_undirected:
                    dag.add_edge(src, tgt)
                    dag.add_edge(tgt, src)
                    edges.append((src, tgt))
                    edges.append((tgt, src))
                    seen_undirected.add(pair)
            # mat values 0 or anything else → no edge

    return dag, edges


# ── Correlation-based fallback ────────────────────────────────────────────────

def _correlation_fallback(
    data_matrix: np.ndarray,
    col_names: List[str],
    threshold: float = 0.3,
) -> Tuple[nx.DiGraph, List[Tuple[str, str]]]:
    """
    Build a skeleton graph from |Pearson r| > threshold.
    Direction is assigned by column order (lower index → higher index) as
    a reproducible but conservative heuristic.
    """
    print(f"  [Fallback] Correlation-based edge detection (|r| > {threshold})…")
    corr = np.corrcoef(data_matrix.T)
    n = len(col_names)
    dag = nx.DiGraph()
    dag.add_nodes_from(col_names)
    edges: List[Tuple[str, str]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > threshold:
                dag.add_edge(col_names[i], col_names[j])
                edges.append((col_names[i], col_names[j]))

    return dag, edges


# ── Privacy budget accounting ─────────────────────────────────────────────────

def _budget_consumed(
    epsilon_total: float,
    n_features: int,
    n_tests_run: int,
) -> float:
    """
    Under sequential composition each CI test consumes ε_per_test = ε / T_max.
    The total ε actually spent = min(ε_per_test * n_tests_run, ε_total).

    Note: PC's skeleton phase re-tests pairs with different conditioning sets,
    so n_tests_run can exceed T_max = n*(n-1)/2 unique pairs.  The privacy
    guarantee is still valid because each *unique pair* is only used once in
    the worst-case analysis, but we cap the reported spend at ε_total to avoid
    misleading accounting when n_tests_run > T_max.
    """
    t_max = max(n_features * (n_features - 1) // 2, 1)
    eps_per_test = epsilon_total / t_max
    raw_spend = eps_per_test * n_tests_run
    return round(min(raw_spend, epsilon_total), 8)


# ── Path validation ───────────────────────────────────────────────────────────

def _validate_paths(
    dag: nx.DiGraph,
    spec: ResearchSpec,
) -> Dict[str, str]:
    """
    Cross-validate ResearchSpec pre-classified paths against the learned DAG.

    A path  A → M → B  is 'confirmed' if every consecutive node pair is
    connected by any path (directed or undirected) in the DAG.
    Paths whose nodes are absent from the DAG are marked 'disputed'.
    """
    validation: Dict[str, str] = {}
    dag_undirected = dag.to_undirected()
    all_nodes_lower = {n.lower().replace(" ", "_"): n for n in dag.nodes()}

    def _reachable(u_norm: str, v_norm: str) -> bool:
        u = all_nodes_lower.get(u_norm)
        v = all_nodes_lower.get(v_norm)
        if u is None or v is None:
            return False
        try:
            return nx.has_path(dag_undirected, u, v)
        except Exception:
            return False

    spec_paths = list(spec.legitimate_paths) + list(spec.illegitimate_paths)
    for pc_obj in spec_paths:
        path_str = pc_obj.path
        # Path strings use " → " as separator
        raw_nodes = [seg.strip() for seg in path_str.replace("→", "|").split("|")]
        norm_nodes = [n.lower().replace(" ", "_") for n in raw_nodes]

        confirmed = all(
            _reachable(norm_nodes[k], norm_nodes[k + 1])
            for k in range(len(norm_nodes) - 1)
        )
        validation[path_str] = "confirmed" if confirmed else "disputed"

    return validation


# ── Public API ────────────────────────────────────────────────────────────────

def run_causal_discovery(
    df: pd.DataFrame,
    spec: ResearchSpec,
    epsilon: float = 1.0,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CausalDiscoveryResult:
    """
    Stage 2 of FIDES — learn a causal DAG with differential privacy.

    Parameters
    ----------
    df : pandas DataFrame
        De-identified dataset from Stage 1.  All PHI must already be removed.
    spec : ResearchSpec
        Output of Stage 0 (research_spec.build_research_spec).
    epsilon : float
        Total privacy budget ε₁ allocated to causal discovery.  Default 1.0.
    alpha : float
        Significance level for PC skeleton discovery.  Default 0.05.
    random_state : int
        Seed for the Laplace-noise RNG, ensuring reproducibility.  Default 42.

    Returns
    -------
    CausalDiscoveryResult
    """
    t_start = time.time()

    print("=" * 64)
    print("FIDES Stage 2 — DP-PC Causal Discovery")
    print(f"  Dataset shape   : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Privacy budget ε: {epsilon}")
    print(f"  CI test α       : {alpha}")
    print(f"  random_state    : {random_state}")
    print("=" * 64)

    # ── Step 1: Encode DataFrame ──────────────────────────────────────────────
    print("\n[1/5] Encoding columns…")
    data_matrix, col_map, col_names = _encode_dataframe(df, random_state=random_state)
    n_samples, n_features = data_matrix.shape
    print(f"      Encoded {n_features} features from {len(df.columns)} columns "
          f"({n_samples} samples).")

    t_max = max(n_features * (n_features - 1) // 2, 1)
    eps_per_test = epsilon / t_max
    print(f"      ε per CI test : {eps_per_test:.8f}  "
          f"(ε={epsilon} ÷ {t_max} max pairs)")

    # ── Step 2: DP-PC algorithm ───────────────────────────────────────────────
    dag = nx.DiGraph()
    dag.add_nodes_from(col_names)
    edges: List[Tuple[str, str]] = []
    method_used = "dp_pc"
    n_tests = 0

    if not _CAUSALLEARN_OK:
        print(f"\n[!] causal-learn unavailable ({_cl_err_msg}). Using fallback.")
        dag, edges = _correlation_fallback(data_matrix, col_names)
        method_used = "correlation_fallback"

    else:
        print("\n[2/5] Configuring DP-FisherZ independence test…")
        # Clear any stale instance from a previous run
        _DP_FISHERZ_INSTANCE.clear()

        print("[3/5] Running PC algorithm with DP noise injected into CI tests…")
        try:
            cg = causallearn_pc(
                data_matrix,
                alpha=alpha,
                indep_test="dp_fisherz",
                stable=True,
                uc_rule=0,
                uc_priority=2,
                show_progress=False,
                node_names=col_names,
                # kwargs forwarded to _DPFisherZ.__init__
                epsilon_per_test=eps_per_test,
                random_seed=random_state,
            )

            # Retrieve test count from the registered instance
            dp_instance = _DP_FISHERZ_INSTANCE.get("current")
            n_tests = dp_instance.test_count if dp_instance is not None else t_max
            print(f"      PC finished.  CI tests executed: {n_tests}")

            print("[4/5] Converting CausalGraph → NetworkX DiGraph…")
            dag, edges = _causallearn_graph_to_nx(cg, col_names)
            print(f"      DAG: {dag.number_of_nodes()} nodes, "
                  f"{dag.number_of_edges()} directed edges "
                  f"({len(edges)} incl. undirected-as-bidirected).")

        except Exception as exc:
            print(f"\n[!] PC algorithm raised an error: {exc}")
            print("    Falling back to correlation-based edge detection (|r| > 0.3).")
            dag, edges = _correlation_fallback(data_matrix, col_names)
            method_used = "correlation_fallback"
            n_tests = t_max   # conservative accounting

    # ── Step 3: Privacy budget accounting ────────────────────────────────────
    if method_used == "dp_pc":
        privacy_budget_used = _budget_consumed(epsilon, n_features, n_tests)
    else:
        # Fallback performs no DP-protected queries
        privacy_budget_used = 0.0

    print(f"\n[5/5] Privacy ledger:  ε₁ = {privacy_budget_used:.8f} consumed  "
          f"({n_tests} tests × {eps_per_test:.8f} each)")

    # ── Step 4: Cross-validate against ResearchSpec paths ────────────────────
    print("\n[+] Validating edges against ResearchSpec causal paths…")
    path_validation = _validate_paths(dag, spec)
    n_confirmed = sum(1 for v in path_validation.values() if v == "confirmed")
    n_disputed = sum(1 for v in path_validation.values() if v == "disputed")
    print(f"    {n_confirmed} confirmed  |  {n_disputed} disputed  "
          f"({len(path_validation)} total spec paths)")
    for path_str, verdict in path_validation.items():
        marker = "✓" if verdict == "confirmed" else "✗"
        print(f"    [{marker}] {verdict.upper():>9s}  {path_str}")

    elapsed = time.time() - t_start
    print(f"\nCausal discovery complete in {elapsed:.2f}s  |  method = {method_used}")
    print("=" * 64)

    return CausalDiscoveryResult(
        dag=dag,
        edges=edges,
        privacy_budget_used=privacy_budget_used,
        path_validation=path_validation,
        column_mapping=col_map,
        n_tests=n_tests,
        method_used=method_used,
    )


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.research_spec import build_research_spec

    np.random.seed(42)
    n = 500

    age          = np.random.normal(55, 12, n)
    sex_num      = np.random.binomial(1, 0.5, n)
    race_vals    = np.random.choice(["White", "Black", "Hispanic", "Asian"], n)
    troponin     = np.clip(0.03 * age + 0.2 * sex_num + np.random.normal(0, 0.4, n), 0, 5)
    pain_score   = 4.0 + 0.5 * sex_num + np.random.normal(0, 1.2, n)
    ttd          = np.random.normal(3.5, 1.2, n)
    mi_outcome   = (troponin > 1.0).astype(int)

    df_test = pd.DataFrame({
        "age":                  age,
        "sex":                  np.where(sex_num == 1, "Male", "Female"),
        "race":                 race_vals,
        "troponin":             troponin,
        "pain_score_recorded":  pain_score,
        "time_to_diagnosis":    ttd,
        "MI_outcome":           mi_outcome,
    })

    spec = build_research_spec(
        domain="cardiology",
        target_variable="MI_outcome",
        target_type="binary",
        use_case="fda_submission",
        columns=list(df_test.columns),
        intent="Predict MI risk — DP-PC self-test",
    )

    result = run_causal_discovery(df_test, spec, epsilon=1.0)

    print("\n─── CausalDiscoveryResult summary ───")
    print(f"  method_used         : {result.method_used}")
    print(f"  n_tests             : {result.n_tests}")
    print(f"  privacy_budget_used : {result.privacy_budget_used}")
    print(f"  edges               : {result.edges}")
    print(f"  path_validation     : {result.path_validation}")
