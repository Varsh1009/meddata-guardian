"""
FIDES Stage 0 — Research Specification Engine
Zero PHI. Runs before any data upload.
Takes column names + research intent, returns ResearchSpec.
"""

import json
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

# ── Pydantic models ──────────────────────────────────────────────────────────

class PathClassification(BaseModel):
    path: str
    classification: Literal["legitimate", "illegitimate", "proxy", "disputed"]
    reason: str

class SubgroupSpec(BaseModel):
    attribute: str
    role: Literal["primary", "secondary", "proxy", "intersection"]
    reason: str
    min_sample_size: int

class ClinicalConstraint(BaseModel):
    feature: str
    constraint_type: Literal["range", "conditional", "logical", "correlation"]
    rule: str

class ResearchSpec(BaseModel):
    domain: str
    target_variable: str
    target_type: Literal["binary", "continuous", "multiclass"]
    use_case: Literal["research", "clinical_deployment", "fda_submission", "irb_audit"]
    required_columns: List[str]
    protected_attributes: List[SubgroupSpec]
    proxy_variables: List[SubgroupSpec]
    relevant_intersections: List[str]
    legitimate_paths: List[PathClassification]
    illegitimate_paths: List[PathClassification]
    clinical_constraints: List[ClinicalConstraint]
    fairness_weights: Dict[str, float]
    warnings: List[str]


# ── Clinical knowledge base (hardcoded domain rules) ─────────────────────────

DOMAIN_KNOWLEDGE = {
    "cardiology": {
        "protected_primary": ["sex", "gender", "race", "ethnicity"],
        "protected_secondary": ["insurance", "zip_code", "income"],
        "proxy_columns": {
            "zip_code": "socioeconomic + race proxy (US healthcare)",
            "insurance": "care access proxy (mediates race → treatment delay)",
            "income": "socioeconomic proxy",
            "referring_source": "care access proxy",
        },
        "legitimate_paths": [
            ("sex", "hormone_levels", "cardiovascular_risk", "Biological: estrogen lowers atherosclerosis rate"),
            ("sex", "symptom_profile", "presentation", "Biological: women present with atypical MI symptoms"),
            ("age", "arterial_stiffness", "blood_pressure", "Biological: aging increases arterial stiffness"),
            ("ancestry", "lipid_metabolism", "LDL", "Biological: genetic ancestry affects lipid processing"),
        ],
        "illegitimate_paths": [
            ("race", "pain_score_recorded", "troponin_ordered", "Discrimination: undertreating pain in Black patients"),
            ("race", "time_to_diagnosis", "disease_severity", "Disparity: delayed diagnosis by race"),
            ("sex", "symptom_dismissed", "treatment_delay", "Discrimination: women's symptoms dismissed"),
            ("zip_code", "ED_wait_time", "treatment_intensity", "Proxy: zip code mediates care access"),
        ],
        "key_intersections": ["sex × race × age_group", "race × insurance"],
        "clinical_constraints": [
            ("troponin", "range", "[0.0, 50.0] ng/mL"),
            ("systolic_bp", "range", "[60, 260] mmHg"),
            ("diastolic_bp", "range", "[40, 160] mmHg"),
            ("age", "range", "[0, 120] years"),
            ("heart_rate", "range", "[20, 300] bpm"),
        ],
        "symptom_sex_mediators": ["fatigue", "nausea", "jaw_pain", "arm_pain"],
    },
    "endocrinology": {
        "protected_primary": ["race", "ethnicity", "sex"],
        "protected_secondary": ["insurance", "zip_code"],
        "proxy_columns": {
            "zip_code": "SES proxy",
            "insurance": "access proxy",
        },
        "legitimate_paths": [
            ("ancestry", "genetic_risk", "T2D_susceptibility", "Biological: genetic T2D risk varies by ancestry"),
            ("sex", "hormones", "insulin_sensitivity", "Biological: hormonal differences affect insulin"),
            ("age", "beta_cell_function", "glucose_regulation", "Biological: beta cell decline with age"),
        ],
        "illegitimate_paths": [
            ("race", "dietary_advice_quality", "glycemic_control", "Disparity: unequal nutrition counseling"),
            ("race", "medication_adherence_assumed", "prescription_intensity", "Discrimination: racial assumptions in prescribing"),
            ("insurance", "follow_up_rate", "complication_detection", "Proxy: insurance mediates follow-up care"),
        ],
        "key_intersections": ["race × sex × age_group", "race × insurance"],
        "clinical_constraints": [
            ("HbA1c", "range", "[3.5, 20.0] %"),
            ("glucose", "range", "[40, 700] mg/dL"),
            ("bmi", "range", "[10.0, 80.0] kg/m²"),
            ("age", "range", "[0, 120] years"),
        ],
        "symptom_sex_mediators": [],
    },
    "oncology": {
        "protected_primary": ["race", "ethnicity", "sex"],
        "protected_secondary": ["insurance", "zip_code", "income"],
        "proxy_columns": {
            "zip_code": "SES + race proxy",
            "insurance": "screening access proxy",
            "income": "SES proxy",
        },
        "legitimate_paths": [
            ("ancestry", "BRCA_prevalence", "cancer_risk", "Biological: BRCA mutations vary by ancestry"),
            ("sex", "hormone_exposure", "cancer_type", "Biological: sex-specific cancers"),
            ("age", "cumulative_exposure", "cancer_risk", "Biological: age is direct cancer risk factor"),
        ],
        "illegitimate_paths": [
            ("race", "screening_rate", "stage_at_diagnosis", "Disparity: unequal cancer screening access"),
            ("insurance", "treatment_intensity", "survival", "Proxy: insurance determines treatment quality"),
            ("zip_code", "specialist_access", "treatment_delay", "Proxy: geography determines specialist access"),
        ],
        "key_intersections": ["race × sex", "race × insurance × age_group"],
        "clinical_constraints": [
            ("age", "range", "[0, 120] years"),
            ("tumor_size", "range", "[0.0, 30.0] cm"),
            ("psa", "range", "[0.0, 500.0] ng/mL"),
        ],
        "symptom_sex_mediators": [],
    },
    "general": {
        "protected_primary": ["race", "ethnicity", "sex", "gender", "age"],
        "protected_secondary": ["insurance", "zip_code"],
        "proxy_columns": {
            "zip_code": "SES + race proxy",
            "insurance": "care access proxy",
        },
        "legitimate_paths": [],
        "illegitimate_paths": [
            ("race", "treatment_quality", "outcome", "Potential disparity in care quality"),
            ("sex", "diagnosis_rate", "outcome", "Potential sex bias in diagnosis"),
        ],
        "key_intersections": ["race × sex × age_group"],
        "clinical_constraints": [
            ("age", "range", "[0, 120] years"),
        ],
        "symptom_sex_mediators": [],
    },
}

# Minimum sample sizes from PAC bounds (conservative estimates from literature)
SUBGROUP_MIN_SAMPLES = {
    "primary_binary": 200,
    "primary_rare": 300,
    "secondary": 150,
    "intersection_2way": 100,
    "intersection_3way": 200,
}

USE_CASE_WEIGHTS = {
    "fda_submission":      {"pathway": 0.35, "statistical": 0.30, "coverage": 0.20, "intersectional": 0.15},
    "clinical_deployment": {"pathway": 0.30, "statistical": 0.30, "coverage": 0.25, "intersectional": 0.15},
    "research":            {"pathway": 0.40, "statistical": 0.20, "coverage": 0.20, "intersectional": 0.20},
    "irb_audit":           {"pathway": 0.25, "statistical": 0.25, "coverage": 0.25, "intersectional": 0.25},
}


# ── Core engine ───────────────────────────────────────────────────────────────

def _normalize_col(col: str) -> str:
    return col.lower().strip().replace(" ", "_")


def _detect_protected_attributes(columns: List[str], domain_info: dict) -> List[SubgroupSpec]:
    norm_cols = [_normalize_col(c) for c in columns]
    specs = []

    primary_keywords = ["race", "ethnicity", "sex", "gender", "race_ethnicity"]
    secondary_keywords = ["age", "age_group", "age_band"]

    for col, norm in zip(columns, norm_cols):
        role = None
        reason = ""
        for kw in primary_keywords:
            if kw in norm:
                role = "primary"
                reason = f"Demographic protected attribute — direct discrimination risk"
                break
        if role is None:
            for kw in secondary_keywords:
                if kw in norm:
                    role = "secondary"
                    reason = "Age is a secondary protected attribute and biological modifier"
                    break
        if role:
            specs.append(SubgroupSpec(
                attribute=col,
                role=role,
                reason=reason,
                min_sample_size=SUBGROUP_MIN_SAMPLES["primary_binary"] if role == "primary"
                                else SUBGROUP_MIN_SAMPLES["secondary"]
            ))
    return specs


def _detect_proxy_variables(columns: List[str], domain_info: dict) -> List[SubgroupSpec]:
    norm_cols = {_normalize_col(c): c for c in columns}
    specs = []
    for norm_key, reason in domain_info.get("proxy_columns", {}).items():
        for col_norm, col_orig in norm_cols.items():
            if norm_key in col_norm:
                specs.append(SubgroupSpec(
                    attribute=col_orig,
                    role="proxy",
                    reason=reason,
                    min_sample_size=0
                ))
    return specs


def _build_path_classifications(columns: List[str], domain_info: dict) -> tuple:
    norm_cols = set(_normalize_col(c) for c in columns)
    legitimate = []
    illegitimate = []

    for path_tuple in domain_info.get("legitimate_paths", []):
        src, med, tgt, reason = path_tuple
        if src in norm_cols or tgt in norm_cols:
            legitimate.append(PathClassification(
                path=f"{src} → {med} → {tgt}",
                classification="legitimate",
                reason=reason
            ))

    for path_tuple in domain_info.get("illegitimate_paths", []):
        src, med, tgt, reason = path_tuple
        illegitimate.append(PathClassification(
            path=f"{src} → {med} → {tgt}",
            classification="illegitimate",
            reason=reason
        ))

    return legitimate, illegitimate


def _build_clinical_constraints(domain_info: dict) -> List[ClinicalConstraint]:
    constraints = []
    for feat, ctype, rule in domain_info.get("clinical_constraints", []):
        constraints.append(ClinicalConstraint(
            feature=feat,
            constraint_type=ctype,
            rule=rule
        ))
    return constraints


def _build_warnings(columns: List[str], domain_info: dict, protected: List[SubgroupSpec]) -> List[str]:
    warnings_list = []
    norm_cols = set(_normalize_col(c) for c in columns)

    for mediator in domain_info.get("symptom_sex_mediators", []):
        if mediator in norm_cols:
            warnings_list.append(
                f"Column '{mediator}' is a sex-specific symptom mediator (legitimate biological path). "
                f"Do NOT block this in generation — it carries real clinical signal."
            )

    if not protected:
        warnings_list.append(
            "No protected attributes detected from column names. "
            "Manually verify: are demographic columns named differently?"
        )

    if "zip_code" in norm_cols or "zipcode" in norm_cols:
        warnings_list.append(
            "zip_code detected — treated as race/SES proxy. "
            "Confirm this is appropriate for your dataset."
        )

    return warnings_list


# ── Public API ────────────────────────────────────────────────────────────────

def build_research_spec(
    domain: str,
    target_variable: str,
    target_type: str,
    use_case: str,
    columns: List[str],
    intent: str = "",
) -> ResearchSpec:
    """
    Stage 0: Build ResearchSpec from column names + intent.
    No data required.
    """
    domain_key = domain.lower()
    if domain_key not in DOMAIN_KNOWLEDGE:
        domain_key = "general"

    domain_info = DOMAIN_KNOWLEDGE[domain_key]

    protected = _detect_protected_attributes(columns, domain_info)
    proxies = _detect_proxy_variables(columns, domain_info)
    legitimate_paths, illegitimate_paths = _build_path_classifications(columns, domain_info)
    constraints = _build_clinical_constraints(domain_info)
    warn = _build_warnings(columns, domain_info, protected)

    # Required columns = protected attrs + proxies + target (strip identifiers)
    identifier_keywords = ["name", "ssn", "mrn", "phone", "email", "address",
                           "dob", "birth", "id", "record", "account"]
    required = [
        c for c in columns
        if not any(kw in _normalize_col(c) for kw in identifier_keywords)
    ]

    intersections = domain_info.get("key_intersections", [])

    weights = USE_CASE_WEIGHTS.get(use_case, USE_CASE_WEIGHTS["research"])

    return ResearchSpec(
        domain=domain,
        target_variable=target_variable,
        target_type=target_type,
        use_case=use_case,
        required_columns=required,
        protected_attributes=protected,
        proxy_variables=proxies,
        relevant_intersections=intersections,
        legitimate_paths=legitimate_paths,
        illegitimate_paths=illegitimate_paths,
        clinical_constraints=constraints,
        fairness_weights=weights,
        warnings=warn,
    )


def spec_summary(spec: ResearchSpec) -> str:
    lines = [
        f"=== FIDES Research Specification ===",
        f"Domain:         {spec.domain}",
        f"Target:         {spec.target_variable} ({spec.target_type})",
        f"Use case:       {spec.use_case}",
        f"",
        f"Protected attributes ({len(spec.protected_attributes)}):",
    ]
    for s in spec.protected_attributes:
        lines.append(f"  [{s.role.upper()}] {s.attribute} — {s.reason}")
    lines.append(f"\nProxy variables ({len(spec.proxy_variables)}):")
    for p in spec.proxy_variables:
        lines.append(f"  {p.attribute} — {p.reason}")
    lines.append(f"\nLegitimate paths ({len(spec.legitimate_paths)}):")
    for p in spec.legitimate_paths:
        lines.append(f"  KEEP: {p.path}")
    lines.append(f"\nIllegitimate paths ({len(spec.illegitimate_paths)}):")
    for p in spec.illegitimate_paths:
        lines.append(f"  BLOCK: {p.path} ({p.reason})")
    lines.append(f"\nClinical constraints: {len(spec.clinical_constraints)} rules loaded")
    lines.append(f"Intersections to monitor: {', '.join(spec.relevant_intersections)}")
    if spec.warnings:
        lines.append(f"\nWARNINGS:")
        for w in spec.warnings:
            lines.append(f"  ⚠ {w}")
    return "\n".join(lines)


if __name__ == "__main__":
    spec = build_research_spec(
        domain="cardiology",
        target_variable="MI_outcome",
        target_type="binary",
        use_case="fda_submission",
        columns=["age", "sex", "race", "zip_code", "insurance",
                 "chest_pain", "fatigue", "nausea", "troponin",
                 "time_to_diagnosis", "pain_score", "MI_outcome"],
        intent="Predict MI risk in adult ED population"
    )
    print(spec_summary(spec))
