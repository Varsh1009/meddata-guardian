"""
FIDES Demo — Run this to see the full pipeline.
Usage: python test_fides.py
"""
import sys, warnings, json
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import pandas as pd

from src.utils.research_spec  import build_research_spec, spec_summary
from src.utils.hipaa_ingestion import ingest
from src.utils.causal_discovery import run_causal_discovery
from src.utils.cds_assessor    import CDSAssessor
from src.utils.intervention_optimizer import optimize_intervention
from src.utils.certificate_builder    import build_certificate

# ── Pick a dataset ────────────────────────────────────────────────────────────
# dataset1 = heart disease with data quality issues
# dataset2 = diabetes with gender bias  (73% male)
# dataset3 = heart disease with zero Indigenous representation
# dataset4 = diabetes combined issues

DATASET  = "data/synthetic/dataset2_diabetes_gender_bias.csv"
DOMAIN   = "endocrinology"   # cardiology | endocrinology | oncology | general
TARGET   = "diabetes"        # last column name in your CSV
USE_CASE = "research"        # research | fda_submission | irb_audit | clinical_deployment

print("\n" + "="*60)
print("  FIDES — Full Pipeline Demo")
print("="*60)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET)
print(f"\nDataset loaded: {df.shape[0]} patients, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"Target distribution:\n{df[TARGET].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0: Research Specification (NO DATA — just column names)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 0 — Research Specification (Zero PHI)")
print("─"*60)

spec = build_research_spec(
    domain         = DOMAIN,
    target_variable= TARGET,
    target_type    = "binary",
    use_case       = USE_CASE,
    columns        = list(df.columns),
    intent         = f"Predict {TARGET} risk in adult population"
)
print(spec_summary(spec))

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: HIPAA Ingestion
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 1 — HIPAA Ingestion")
print("─"*60)

clean_df, audit = ingest(df, spec, user_id="demo_user", irb_protocol="IRB-2024-DEMO")
print(f"\nAudit record: {json.dumps({k:v for k,v in audit.items() if k != 'original_hash'}, indent=2)}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Causal Discovery
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 2 — Causal Discovery (DP-PC Algorithm)")
print("─"*60)

causal = run_causal_discovery(clean_df, spec, epsilon=1.0)
print(f"\nCausal edges found: {len(causal.edges)}")
for src, tgt in causal.edges[:10]:
    print(f"  {src}  →  {tgt}")
if len(causal.edges) > 10:
    print(f"  ... and {len(causal.edges)-10} more")
print(f"\nPath validation:")
for path, status in causal.path_validation.items():
    print(f"  [{status.upper()}]  {path}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: CDS Assessment
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 3 — Four-Condition CDS Assessment")
print("─"*60)

assessor = CDSAssessor(clean_df, spec, causal)
cds = assessor.assess()

print(f"""
  ┌─────────────────────────────────────────────┐
  │          CDS SCORECARD                      │
  ├─────────────────────────────────────────────┤
  │  Pathway Sufficiency:    {cds.condition_scores.get('pathway',0):.3f}               │
  │  Statistical Sufficiency:{cds.condition_scores.get('statistical',0):.3f}               │
  │  Phenotypic Coverage:    {cds.condition_scores.get('coverage',0):.3f}               │
  │  Intersectional:         {cds.condition_scores.get('intersectional',0):.3f}               │
  ├─────────────────────────────────────────────┤
  │  OVERALL CDS SCORE:      {cds.cds_score:.3f}               │
  │  95% CI: [{cds.confidence_interval[0]:.2f}, {cds.confidence_interval[1]:.2f}]                  │
  │  Threshold: {0.75 if USE_CASE=='research' else 0.85:.2f}  → {'PASS ✓' if cds.threshold_met else 'FAIL ✗'}                  │
  └─────────────────────────────────────────────┘""")

if cds.insufficiency_masking_flags:
    print("\n  ⚠  INSUFFICIENCY MASKING WARNINGS:")
    for flag in cds.insufficiency_masking_flags[:3]:
        print(f"    {flag[:100]}...")

print("\n  Recommendations:")
for rec in cds.recommendations:
    print(f"    • {rec}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: Minimum Intervention Optimizer
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 4 — Minimum Intervention Plan (MILP)")
print("─"*60)

plan = optimize_intervention(cds, spec)

print(f"\n  Total new patients needed: {plan.total_new_patients}")
print(f"  Estimated recruitment cost: ${plan.estimated_cost_usd:,}")
print(f"  Solver status: {plan.solver_status}")
print(f"\n  Recruitment targets:")
for g in plan.groups:
    print(f"\n    Group: {g.group_name}")
    print(f"    Need:  {g.n_required} patients")
    print(f"    Why:   {g.condition_driving_need}")
    print(f"    Sites: {', '.join(g.recommended_sites[:2])}")
    if g.phenotype_profile:
        for feat, vals in list(g.phenotype_profile.items())[:2]:
            print(f"    {feat}: {vals}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: Equitable-Care Generation (optional — uncomment to run)
# ─────────────────────────────────────────────────────────────────────────────
# print("\n" + "─"*60)
# print("STAGE 5 — Equitable-Care Generation")
# print("─"*60)
# from src.utils.equitable_generator import EquitableGenerator
# generator = EquitableGenerator(spec, causal, epsilon=0.5)
# gen = generator.generate(clean_df, plan)
# print(f"  Generated: {gen.n_generated} patients")
# print(f"  Rejected (invalid): {gen.n_rejected}")
# print(f"  Re-id risk: {gen.reidentification_risk:.4f}  (<0.09 = safe)")
# print(f"  k-anonymity: k={gen.k_anonymity_achieved}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6: Certification
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 6 — Deployment Fitness Certificate")
print("─"*60)

cert = build_certificate(
    dataset_hash    = audit['original_hash'],
    spec_id         = f"{DOMAIN}_{TARGET}",
    use_case        = USE_CASE,
    audit_record    = audit,
    cds_result_before = cds,
    cds_result_after  = cds,
    gen_result        = None,
    intervention_plan = plan,
    irb_protocol      = "IRB-2024-DEMO",
    output_dir        = "/tmp",
)

print(f"""
  ╔══════════════════════════════════════════════╗
  ║  FIDES DEPLOYMENT FITNESS CERTIFICATE        ║
  ╠══════════════════════════════════════════════╣
  ║  Cert ID:   {cert.cert_id[:20]}...   ║
  ║  Verdict:   {cert.verdict:<35}║
  ║  CDS Score: {cert.cds_score_after:.3f} / 1.000                        ║
  ╠══════════════════════════════════════════════╣
  ║  HIPAA Compliant:   {'YES ✓' if cert.hipaa_compliant else 'NO ✗':<29}║
  ║  FDA SaMD:          {'YES ✓' if cert.fda_samd_compliant else 'NO ✗':<29}║
  ║  EU AI Act Art.10:  {'YES ✓' if cert.eu_ai_act_compliant else 'NO ✗':<29}║
  ╠══════════════════════════════════════════════╣
  ║  DP Guarantee: ε={cert.dp_epsilon:.3f}, δ=1e-6              ║
  ║  Re-id Risk:   {cert.reidentification_risk:.4f} (<0.09 threshold)       ║
  ╚══════════════════════════════════════════════╝""")

if cert.conditions:
    print("\n  Conditions for full approval:")
    for c in cert.conditions:
        print(f"    • {c}")

print(f"\n  Saved to: /tmp/fides_cert_{cert.cert_id[:12]}.pdf")
print(f"  Signature: {cert.signature[:50]}...")
print("\n" + "="*60)
print("  FIDES PIPELINE COMPLETE")
print("="*60 + "\n")
