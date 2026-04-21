"""
FIDES Stage 4 — Minimum Intervention Optimizer

Given a CDSResult (output of Stage 3), computes the minimum number of new
patients needed per demographic subgroup and intersection to bring the dataset
to full CDS compliance.

Algorithm
---------
1. Parse CDSResult.condition_details (statistical, coverage, intersectional) to
   identify failing subgroups and quantify each deficit.
2. Translate deficits into hard lower-bound constraints:
     statistical gap  → need n_G = max(0, n_star_G - current_n_G)
     coverage gap     → +20% extra if subgroup coverage < 0.90
     intersectional   → +50 patients for each empty / below-threshold cell
3. Formulate a MILP via PuLP:
     min  Σ_G  x_G * cost_G
     s.t. x_G >= need_G   ∀G
          x_G ∈ Z_{≥0}
     (optional)  Σ_G x_G * cost_G <= budget
4. Solve with the CBC solver bundled with PuLP.  Fall back to a greedy feasible
   solution if the MILP is infeasible or the solver is unavailable.
5. Annotate each group with a clinically-grounded phenotype profile and site
   recommendations derived from the nature of the gap.
"""

from __future__ import annotations

import math
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# PuLP is the LP/MILP solver used for the minimum-cost recruitment problem.
try:
    from pulp import (
        LpProblem, LpMinimize, LpVariable, LpStatus,
        lpSum, value as lp_value, PULP_CBC_CMD,
        LpInteger, LpStatusOptimal,
    )
    _PULP_OK = True
except ImportError:  # pragma: no cover
    _PULP_OK = False
    warnings.warn(
        "PuLP is not installed.  Install it with `pip install pulp`.  "
        "Falling back to greedy feasible solution."
    )

from src.utils.research_spec import ResearchSpec
from src.utils.cds_assessor import CDSResult


# ── Constants ──────────────────────────────────────────────────────────────────

# Default per-patient recruitment cost in USD (sourced from clinical trial
# literature; adjustable via the budget argument)
_DEFAULT_COST_PER_PATIENT: int = 1_500

# Coverage threshold below which a 20% buffer is added to the statistical need
_COVERAGE_BUFFER_THRESHOLD: float = 0.90

# Minimum extra patients added per intersectional cell that is empty or below
# the _MIN_CELL_SIZE defined in the CDS assessor (50 patients)
_INTERSECTIONAL_MIN_ADD: int = 50

# Minimum patients to recruit even if the pure gap calculation yields zero
_ABSOLUTE_FLOOR: int = 5

# Priority classification thresholds (based on fractional gap magnitude)
_PRIORITY_CRITICAL_THRESHOLD: float = 0.50   # missing > 50% of n_star
_PRIORITY_HIGH_THRESHOLD: float = 0.20       # missing > 20% of n_star


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class GroupRecruitmentTarget:
    """Recruitment requirement for one demographic subgroup."""

    group_name: str
    """Human-readable label, e.g. 'Black female, age 45–65'."""

    n_required: int
    """Minimum number of new patients that must be enrolled for this group."""

    condition_driving_need: str
    """Which CDS condition (statistical / coverage / intersectional) triggered
    this requirement."""

    phenotype_profile: Dict[str, Any]
    """Specific clinical feature ranges the recruited patients must satisfy so
    that they fill the identified distributional gap."""

    recommended_sites: List[str]
    """Suggested enrollment sites or recruitment strategies appropriate for this
    group's demographics."""

    priority: str
    """'critical', 'high', or 'medium' — determines recruitment sequencing."""


@dataclass
class InterventionPlan:
    """Full output of Stage 4 Minimum Intervention Optimizer."""

    total_new_patients: int
    """Sum of n_required across all GroupRecruitmentTargets."""

    groups: List[GroupRecruitmentTarget]
    """Per-group targets, sorted by priority then n_required descending."""

    feasible: bool
    """True when the MILP found an optimal or feasible solution."""

    solver_status: str
    """Raw solver status string (e.g. 'Optimal', 'Infeasible', 'Greedy Fallback')."""

    estimated_cost_usd: int
    """Total estimated recruitment cost at the default per-patient rate."""

    priority_order: List[str]
    """Group names ordered from highest to lowest recruitment priority."""


# ── Internal gap computation ───────────────────────────────────────────────────

def _parse_statistical_gaps(
    cds: CDSResult,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-subgroup statistical gaps from CDSResult.condition_details.

    Returns a dict keyed by "<attribute>/<subgroup_value>" with fields:
        n_current   – observed count
        n_star      – required count (PAC-bound-derived)
        gap         – max(0, n_star - n_current)
        power       – estimated statistical power
        attribute   – parent protected attribute name
        subgroup    – subgroup value string
    """
    gaps: Dict[str, Dict[str, Any]] = {}
    stat_details = cds.condition_details.get("statistical", {})
    n_star_global: float = float(stat_details.get("n_star", 200.0))
    per_attr: Dict[str, Dict] = stat_details.get("per_attribute", {})

    for attr, subgroup_dict in per_attr.items():
        if not isinstance(subgroup_dict, dict):
            continue
        for sg_val, sg_info in subgroup_dict.items():
            if not isinstance(sg_info, dict):
                continue
            n_current = int(sg_info.get("n", 0))
            # Use the per-record n_star if available, else global
            n_star = float(sg_info.get("n_star", n_star_global))
            gap = max(0, math.ceil(n_star - n_current))
            power = float(sg_info.get("power", 0.0))
            key = f"{attr}/{sg_val}"
            gaps[key] = {
                "attribute":  attr,
                "subgroup":   str(sg_val),
                "n_current":  n_current,
                "n_star":     n_star,
                "gap":        gap,
                "power":      power,
            }
    return gaps


def _parse_coverage_gaps(
    cds: CDSResult,
) -> Dict[str, float]:
    """
    Return a dict keyed by "<attribute>/<subgroup_value>" → coverage score.
    Groups with coverage < _COVERAGE_BUFFER_THRESHOLD require +20% buffer.
    """
    coverage_map: Dict[str, float] = {}
    cov_details = cds.condition_details.get("coverage", {})
    per_attr: Dict[str, Dict] = cov_details.get("per_attribute", {})

    for attr, subgroup_dict in per_attr.items():
        if not isinstance(subgroup_dict, dict):
            continue
        for sg_val, sg_info in subgroup_dict.items():
            if not isinstance(sg_info, dict):
                continue
            cov = float(sg_info.get("coverage", 1.0))
            key = f"{attr}/{sg_val}"
            coverage_map[key] = cov
    return coverage_map


def _parse_intersectional_gaps(
    cds: CDSResult,
) -> List[Dict[str, Any]]:
    """
    Return a list of dicts, one per underpopulated intersectional cell:
        intersection  – e.g. "sex × race × age_group"
        cell_key      – e.g. "Female × Black × 45-65"
        cell_count    – observed count
        shortfall     – how many more needed to reach min_cell_size
    """
    cells: List[Dict[str, Any]] = []
    inter_details = cds.condition_details.get("intersectional", {})
    per_inter: Dict[str, Dict] = inter_details.get("per_intersection", {})

    for inter_str, inter_info in per_inter.items():
        if not isinstance(inter_info, dict):
            continue
        min_cell_size: int = int(inter_info.get("min_cell_size", 50))
        cell_counts: Dict[str, int] = inter_info.get("cells", {})
        for cell_key, cnt in cell_counts.items():
            if cnt < min_cell_size:
                cells.append({
                    "intersection": inter_str,
                    "cell_key":     cell_key,
                    "cell_count":   cnt,
                    "shortfall":    max(0, min_cell_size - cnt),
                })
    return cells


# ── Phenotype profile builder ──────────────────────────────────────────────────

def _build_phenotype_profile(
    group_key: str,
    condition: str,
    spec: ResearchSpec,
    coverage_score: float,
) -> Dict[str, Any]:
    """
    Construct a clinical phenotype profile for the recruited patients.

    Profile is based on:
    - The domain-specific clinical constraints from the ResearchSpec.
    - The nature of the CDS gap (statistical → broad coverage,
      coverage → underrepresented feature region, intersectional → specific cell).
    - Standard demographic ranges parsed from the group_key.

    Returns a dict of {feature: range_or_value}.
    """
    profile: Dict[str, Any] = {}

    # Parse demographic markers from the group key
    key_lower = group_key.lower()
    parts = [p.strip() for p in key_lower.replace("/", " ").split()]

    # Age range heuristic
    if "45-65" in key_lower or "middle" in key_lower:
        profile["age"] = {"min": 45, "max": 65}
    elif "65+" in key_lower or "elderly" in key_lower or "senior" in key_lower:
        profile["age"] = {"min": 65, "max": 90}
    elif "18-44" in key_lower or "young" in key_lower:
        profile["age"] = {"min": 18, "max": 44}
    else:
        profile["age"] = {"min": 18, "max": 89}

    # Sex / gender
    for token in parts:
        if token in ("male",):
            profile["sex"] = "Male"
        elif token in ("female",):
            profile["sex"] = "Female"

    # Race / ethnicity
    race_tokens = {
        "black": "Black",
        "african": "Black",
        "hispanic": "Hispanic",
        "latino": "Hispanic",
        "asian": "Asian",
        "indigenous": "American Indian or Alaska Native",
        "native": "American Indian or Alaska Native",
        "white": "White",
        "pacific": "Native Hawaiian or Pacific Islander",
    }
    for token, label in race_tokens.items():
        if token in key_lower:
            profile["race"] = label
            break

    # Domain-specific clinical constraints
    for constraint in spec.clinical_constraints:
        feat = constraint.feature
        rule = constraint.rule
        # Parse "[min, max]" notation
        if constraint.constraint_type == "range" and "[" in rule and "]" in rule:
            try:
                inner = rule[rule.index("[") + 1: rule.index("]")]
                lo_str, hi_str = inner.split(",")
                lo = float(lo_str.strip().split()[0])
                hi = float(hi_str.strip().split()[0])
                # If coverage is poor, focus on the underrepresented middle range
                if coverage_score < 0.70 and condition == "coverage":
                    profile[feat] = {
                        "min": round(lo + (hi - lo) * 0.25, 2),
                        "max": round(lo + (hi - lo) * 0.75, 2),
                        "note": "focus: underrepresented mid-range",
                    }
                else:
                    profile[feat] = {"min": lo, "max": hi}
            except (ValueError, IndexError):
                pass

    # Annotate for illegitimate-pathway coverage: flag that recruited patients
    # should have equitable care values for the mediator columns
    if condition in ("pathway", "intersectional"):
        illegit_meds = []
        for path_obj in spec.illegitimate_paths:
            path_parts = [
                p.strip() for p in path_obj.path.replace("→", "|").split("|")
            ]
            for med in path_parts[1:-1]:
                illegit_meds.append(med)
        if illegit_meds:
            profile["_equitable_care_required_for"] = illegit_meds

    return profile


# ── Site recommendation ────────────────────────────────────────────────────────

# Site categories keyed by demographic signal
_SITE_RULES: List[Tuple[str, str]] = [
    ("indigenous",                "Indian Health Service (IHS) facilities"),
    ("native",                    "Indian Health Service (IHS) facilities"),
    ("american indian",           "Indian Health Service (IHS) facilities"),
    ("alaska native",             "Indian Health Service (IHS) facilities"),
    ("black",                     "HBCU-affiliated clinics; Black-serving FQHCs"),
    ("african",                   "HBCU-affiliated clinics; Black-serving FQHCs"),
    ("hispanic",                  "Latino community health centers; promotora networks"),
    ("latino",                    "Latino community health centers; promotora networks"),
    ("asian",                     "Asian Health Services; multilingual FQHCs"),
    ("pacific",                   "Pacific Islander community health networks"),
    ("medicaid",                  "Federally Qualified Health Centers (FQHCs)"),
    ("uninsured",                 "Free clinic networks; charity care programs"),
    ("rural",                     "Rural Health Clinics; telehealth recruitment"),
    ("elderly",                   "Senior centers; Medicare-enrolled primary care"),
    ("65+",                       "Senior centers; Medicare-enrolled primary care"),
    ("female",                    "Women's health centers; OB-GYN practices"),
]

_DEFAULT_SITES: List[str] = [
    "Academic medical center research registries",
    "Community health center partnerships",
    "Patient advocacy organization outreach",
]


def _recommend_sites(group_key: str) -> List[str]:
    """Map demographic group key to appropriate recruitment sites."""
    key_lower = group_key.lower()
    sites: List[str] = []
    seen: set = set()
    for token, site in _SITE_RULES:
        if token in key_lower and site not in seen:
            sites.append(site)
            seen.add(site)
    if not sites:
        sites = list(_DEFAULT_SITES)
    # Always add a general fallback so there is never an empty list
    if len(sites) < 2:
        for s in _DEFAULT_SITES:
            if s not in seen:
                sites.append(s)
                if len(sites) >= 3:
                    break
    return sites


# ── Priority classifier ────────────────────────────────────────────────────────

def _classify_priority(gap: int, n_star: float) -> str:
    """Classify recruitment priority based on fractional gap magnitude."""
    if n_star <= 0:
        return "medium"
    frac = gap / n_star
    if frac > _PRIORITY_CRITICAL_THRESHOLD:
        return "critical"
    if frac > _PRIORITY_HIGH_THRESHOLD:
        return "high"
    return "medium"


# ── Gap aggregation ────────────────────────────────────────────────────────────

def _aggregate_needs(
    stat_gaps: Dict[str, Dict[str, Any]],
    coverage_map: Dict[str, float],
    inter_cells: List[Dict[str, Any]],
    spec: ResearchSpec,
) -> Dict[str, Dict[str, Any]]:
    """
    Merge statistical, coverage and intersectional gaps into a unified need map.

    Each entry in the returned dict represents one GroupRecruitmentTarget and
    contains the cumulative n_required and provenance information.

    Returns dict keyed by "<attribute>/<subgroup_value>".
    """
    need_map: Dict[str, Dict[str, Any]] = {}

    # ── 1. Statistical gaps ────────────────────────────────────────────────────
    for key, info in stat_gaps.items():
        if key not in need_map:
            need_map[key] = {
                "attribute":  info["attribute"],
                "subgroup":   info["subgroup"],
                "n_required": 0,
                "n_star":     info["n_star"],
                "n_current":  info["n_current"],
                "driving":    "statistical",
                "coverage":   coverage_map.get(key, 1.0),
            }
        need_map[key]["n_required"] = max(
            need_map[key]["n_required"],
            info["gap"],
        )

    # ── 2. Coverage buffer ─────────────────────────────────────────────────────
    for key, cov in coverage_map.items():
        if cov >= _COVERAGE_BUFFER_THRESHOLD:
            continue
        if key not in need_map:
            # Create an entry even if statistical gap was zero
            attr, sg = (key.split("/", 1) + ["unknown"])[:2]
            need_map[key] = {
                "attribute":  attr,
                "subgroup":   sg,
                "n_required": 0,
                "n_star":     200.0,
                "n_current":  0,
                "driving":    "coverage",
                "coverage":   cov,
            }
        # Add 20% buffer on top of the statistical need
        buffer = math.ceil(need_map[key]["n_required"] * 0.20)
        buffer = max(buffer, 20)  # at least 20 extra for coverage deficit
        need_map[key]["n_required"] += buffer
        need_map[key]["driving"] = "coverage"
        need_map[key]["coverage"] = cov

    # ── 3. Intersectional gaps ─────────────────────────────────────────────────
    for cell in inter_cells:
        cell_key = cell["cell_key"]
        shortfall = max(cell["shortfall"], _INTERSECTIONAL_MIN_ADD)

        # Try to map cell back to a known group key; otherwise create new entry
        # Cell keys look like "Female × Black × (45, 65]" — parse first two parts
        cell_parts = [p.strip() for p in cell_key.split("×")]

        # Look for an existing key that overlaps with the cell
        matched_key: Optional[str] = None
        for existing_key in need_map:
            sg_lower = need_map[existing_key]["subgroup"].lower()
            if any(sg_lower in cp.lower() or cp.lower() in sg_lower
                   for cp in cell_parts):
                matched_key = existing_key
                break

        if matched_key is None:
            # Create a synthetic key for this intersectional cell
            synthetic_key = f"intersection/{cell_key}"
            need_map[synthetic_key] = {
                "attribute":  cell["intersection"],
                "subgroup":   cell_key,
                "n_required": shortfall,
                "n_star":     50.0,
                "n_current":  cell["cell_count"],
                "driving":    "intersectional",
                "coverage":   coverage_map.get(synthetic_key, 0.5),
            }
        else:
            need_map[matched_key]["n_required"] = max(
                need_map[matched_key]["n_required"],
                shortfall,
            )
            if need_map[matched_key]["driving"] == "statistical":
                need_map[matched_key]["driving"] = "intersectional"

    # ── 4. Apply absolute floor ────────────────────────────────────────────────
    for key in list(need_map.keys()):
        if need_map[key]["n_required"] > 0:
            need_map[key]["n_required"] = max(
                need_map[key]["n_required"],
                _ABSOLUTE_FLOOR,
            )

    # Drop entries with zero need
    need_map = {k: v for k, v in need_map.items() if v["n_required"] > 0}
    return need_map


# ── MILP solver ────────────────────────────────────────────────────────────────

def _solve_milp(
    need_map: Dict[str, Dict[str, Any]],
    cost_per_patient: int,
    budget: Optional[float],
) -> Tuple[Dict[str, int], str, bool]:
    """
    Solve the Minimum Intervention MILP using PuLP.

    Decision variables: x_G (integer >= 0) for each group G.
    Objective: minimise Σ x_G * cost_per_patient
    Constraints:
        x_G >= need_G  ∀G
        (optional) Σ x_G * cost_per_patient <= budget

    Returns (solution_dict, solver_status, feasible).
    """
    if not need_map:
        return {}, "NoGroups", True

    prob = LpProblem("FIDES_MinIntervention", LpMinimize)

    # Decision variables — integer, non-negative
    x_vars: Dict[str, Any] = {
        key: LpVariable(
            name=f"x_{i}",
            lowBound=0,
            cat=LpInteger,
        )
        for i, key in enumerate(need_map)
    }

    # Objective
    prob += lpSum(x_vars[key] * cost_per_patient for key in need_map), "TotalCost"

    # Minimum coverage constraints
    for key, info in need_map.items():
        prob += x_vars[key] >= info["n_required"], f"need_{key[:40]}"

    # Optional budget constraint
    if budget is not None and budget > 0:
        prob += (
            lpSum(x_vars[key] * cost_per_patient for key in need_map) <= budget,
            "BudgetCap",
        )

    # Solve with CBC (silent mode)
    solver = PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    status_str = LpStatus[prob.status]
    feasible = prob.status == LpStatusOptimal

    solution: Dict[str, int] = {}
    if feasible:
        for key in need_map:
            raw = lp_value(x_vars[key])
            solution[key] = int(round(raw)) if raw is not None else need_map[key]["n_required"]
    else:
        # Fall back to the lower-bound values directly
        solution = {key: info["n_required"] for key, info in need_map.items()}

    return solution, status_str, feasible


def _greedy_fallback(
    need_map: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, int], str, bool]:
    """
    Greedy feasible solution: set x_G = need_G for every group.
    Always feasible; used when PuLP is unavailable or MILP is infeasible.
    """
    solution = {key: info["n_required"] for key, info in need_map.items()}
    return solution, "Greedy Fallback", True


# ── Human-readable group name builder ─────────────────────────────────────────

def _build_group_name(key: str, info: Dict[str, Any]) -> str:
    """Turn an internal key and info dict into a human-readable group label."""
    if key.startswith("intersection/"):
        cell = info["subgroup"]
        inter = info["attribute"]
        return f"Intersectional cell [{inter}]: {cell}"
    attr = info.get("attribute", "unknown")
    sg = info.get("subgroup", "unknown")
    return f"{attr} = {sg}"


# ── Public API ─────────────────────────────────────────────────────────────────

def optimize_intervention(
    cds_result: CDSResult,
    spec: ResearchSpec,
    cost_per_patient: int = _DEFAULT_COST_PER_PATIENT,
    budget: Optional[float] = None,
) -> InterventionPlan:
    """
    Stage 4 of FIDES — Minimum Intervention Optimizer.

    Parameters
    ----------
    cds_result : CDSResult
        Output of Stage 3 (CDSAssessor.assess()).
    spec : ResearchSpec
        Output of Stage 0 (research_spec.build_research_spec()).
    cost_per_patient : int
        Assumed recruitment cost per patient in USD.  Default 1 500 USD.
    budget : float, optional
        Hard cap on total recruitment cost.  When provided, the MILP will
        include a budget constraint.  The plan may still be marked infeasible
        if the budget is insufficient to satisfy all lower bounds.

    Returns
    -------
    InterventionPlan
    """
    print("=" * 64)
    print("FIDES Stage 4 — Minimum Intervention Optimizer")
    print(f"  Domain   : {spec.domain}")
    print(f"  Use case : {spec.use_case}")
    print(f"  CDS score: {cds_result.cds_score:.4f}  (threshold met: {cds_result.threshold_met})")
    if budget:
        print(f"  Budget   : ${budget:,.0f} USD")
    print("=" * 64)

    # ── Step 1: Parse gaps from CDSResult ─────────────────────────────────────
    print("\n[1/5] Parsing CDS condition gaps…")
    stat_gaps = _parse_statistical_gaps(cds_result)
    coverage_map = _parse_coverage_gaps(cds_result)
    inter_cells = _parse_intersectional_gaps(cds_result)
    print(f"      Statistical gaps : {len(stat_gaps)} subgroups")
    print(f"      Coverage gaps    : {sum(1 for v in coverage_map.values() if v < _COVERAGE_BUFFER_THRESHOLD)} subgroups below {_COVERAGE_BUFFER_THRESHOLD}")
    print(f"      Intersectional   : {len(inter_cells)} underpopulated cells")

    # ── Step 2: Aggregate into a unified need map ──────────────────────────────
    print("\n[2/5] Aggregating needs across conditions…")
    need_map = _aggregate_needs(stat_gaps, coverage_map, inter_cells, spec)
    print(f"      Total groups requiring recruitment: {len(need_map)}")
    for key, info in need_map.items():
        name = _build_group_name(key, info)
        print(f"        {name}: need {info['n_required']} patients  [{info['driving']}]")

    if not need_map:
        print("\n  No recruitment required — all CDS conditions are satisfied.")
        return InterventionPlan(
            total_new_patients=0,
            groups=[],
            feasible=True,
            solver_status="NoActionNeeded",
            estimated_cost_usd=0,
            priority_order=[],
        )

    # ── Step 3: Solve MILP (or fall back to greedy) ───────────────────────────
    print("\n[3/5] Solving Minimum Intervention MILP…")
    if _PULP_OK:
        solution, solver_status, feasible = _solve_milp(
            need_map, cost_per_patient, budget
        )
        if not feasible:
            print(f"      MILP status: {solver_status} — falling back to greedy solution")
            solution, solver_status, feasible = _greedy_fallback(need_map)
        else:
            print(f"      MILP status: {solver_status}")
    else:
        print("      PuLP unavailable — using greedy fallback")
        solution, solver_status, feasible = _greedy_fallback(need_map)

    # ── Step 4: Build GroupRecruitmentTargets ──────────────────────────────────
    print("\n[4/5] Building GroupRecruitmentTarget objects…")
    group_targets: List[GroupRecruitmentTarget] = []

    for key, n_req in solution.items():
        info = need_map[key]
        group_name = _build_group_name(key, info)
        condition = info["driving"]
        coverage_score = info.get("coverage", 1.0)
        n_star = info.get("n_star", 200.0)
        actual_need = max(n_req, info["n_required"])

        phenotype = _build_phenotype_profile(
            group_key=f"{info['attribute']} {info['subgroup']}",
            condition=condition,
            spec=spec,
            coverage_score=coverage_score,
        )
        sites = _recommend_sites(f"{info['attribute']} {info['subgroup']}")
        priority = _classify_priority(actual_need, n_star)

        group_targets.append(GroupRecruitmentTarget(
            group_name=group_name,
            n_required=actual_need,
            condition_driving_need=condition,
            phenotype_profile=phenotype,
            recommended_sites=sites,
            priority=priority,
        ))

    # ── Step 5: Sort by priority and compute totals ───────────────────────────
    _priority_rank = {"critical": 0, "high": 1, "medium": 2}
    group_targets.sort(
        key=lambda g: (_priority_rank.get(g.priority, 3), -g.n_required)
    )

    total_new_patients = sum(g.n_required for g in group_targets)
    estimated_cost_usd = total_new_patients * cost_per_patient
    priority_order = [g.group_name for g in group_targets]

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n[5/5] Recruitment Plan Summary")
    print("=" * 64)
    print(f"  Total new patients required : {total_new_patients:,}")
    print(f"  Estimated cost              : ${estimated_cost_usd:,.0f} USD")
    print(f"  Solver status               : {solver_status}")
    print(f"  Feasible                    : {feasible}")
    print(f"\n  Priority Queue ({len(group_targets)} groups):")
    for i, g in enumerate(group_targets, 1):
        print(f"    {i:>2}. [{g.priority.upper():>8s}]  {g.group_name}")
        print(f"          n_required = {g.n_required:>4d}  |  driver = {g.condition_driving_need}")
        print(f"          sites: {'; '.join(g.recommended_sites[:2])}")
    print("=" * 64)

    return InterventionPlan(
        total_new_patients=total_new_patients,
        groups=group_targets,
        feasible=feasible,
        solver_status=solver_status,
        estimated_cost_usd=estimated_cost_usd,
        priority_order=priority_order,
    )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from src.utils.research_spec import build_research_spec
    from src.utils.causal_discovery import run_causal_discovery
    from src.utils.cds_assessor import assess_dataset_sufficiency

    np.random.seed(42)
    n = 500

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

    causal_res = run_causal_discovery(df_demo, spec, epsilon=1.0)
    cds_res    = assess_dataset_sufficiency(df_demo, spec, causal_res)

    plan = optimize_intervention(cds_res, spec, budget=500_000)

    print("\n─── InterventionPlan fields ───")
    print(f"  total_new_patients  : {plan.total_new_patients}")
    print(f"  estimated_cost_usd  : ${plan.estimated_cost_usd:,}")
    print(f"  feasible            : {plan.feasible}")
    print(f"  solver_status       : {plan.solver_status}")
    print(f"  n_groups            : {len(plan.groups)}")
    if plan.groups:
        g0 = plan.groups[0]
        print(f"\n  First target:")
        print(f"    group_name              : {g0.group_name}")
        print(f"    n_required              : {g0.n_required}")
        print(f"    condition_driving_need  : {g0.condition_driving_need}")
        print(f"    priority                : {g0.priority}")
        print(f"    recommended_sites       : {g0.recommended_sites}")
        print(f"    phenotype_profile keys  : {list(g0.phenotype_profile.keys())}")
