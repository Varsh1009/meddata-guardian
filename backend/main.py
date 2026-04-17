"""
MedGuard AI - FastAPI Backend
Same functionality as Streamlit app; used by React frontend.
Run from project root: uvicorn backend.main:app --reload --app-dir .
Or: cd backend && uvicorn main:app --reload (with path fix below)
"""

import sys
from pathlib import Path

# Add project root so we can import src
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import json
import os
from typing import Any, Dict, List, Optional

import ollama
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import after path is set
from src.utils.phi_scanner import PHIScanner
from src.utils.synthetic_generator import SyntheticDataGenerator
from src.utils.data_quality import DataQualityAnalyzer
from src.utils.bias_detection import BiasDetector
from src.agents.deployment_strategist import DeploymentStrategist

app = FastAPI(title="MedGuard AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_fairness_specialist = None


def _get_fairness_specialist():
    global _fairness_specialist
    if _fairness_specialist is None:
        from src.agents.fairness_specialist import FairnessSpecialist

        _fairness_specialist = FairnessSpecialist()
    return _fairness_specialist


# ---------------------------------------------------------------------------
# Helpers: DataFrame <-> JSON
# ---------------------------------------------------------------------------

def df_to_json(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame to list of dicts (JSON-serializable)."""
    raw = df.replace({np.nan: None}).to_dict(orient="records")
    return _make_serializable(raw)

def json_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """List of dicts to DataFrame."""
    return pd.DataFrame(rows)

def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj

# ---------------------------------------------------------------------------
# Demo datasets
# ---------------------------------------------------------------------------

DEMO_FILES = {
    "Demo 1: Heart Disease (Quality Issues)": "data/synthetic/dataset1_heart_disease_quality.csv",
    "Demo 2: Diabetes (Gender Bias)": "data/synthetic/dataset2_diabetes_gender_bias.csv",
    "Demo 3: Heart Disease (Indigenous Bias)": "data/synthetic/dataset3_heart_disease_indigenous.csv",
    "Demo 4: Combined Problems": "data/synthetic/dataset4_diabetes_combined.csv",
}

def get_demo_path(name: str) -> Optional[str]:
    path = DEMO_FILES.get(name)
    if path and os.path.exists(path):
        return path
    # Try from backend folder
    alt = ROOT / path
    if alt.exists():
        return str(alt)
    return None

# ---------------------------------------------------------------------------
# API: PHI Scan
# ---------------------------------------------------------------------------

@app.post("/api/phi-scan")
async def phi_scan(file: UploadFile = File(...)):
    """Scan uploaded CSV for PHI. Returns is_safe and violations."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "CSV file required")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Invalid CSV: {e}")
    scanner = PHIScanner()
    is_safe, violations = scanner.scan_dataset(df)
    return {"is_safe": is_safe, "violations": violations, "rows": len(df), "columns": len(df.columns)}

@app.post("/api/phi-scan-demo")
async def phi_scan_demo(demo_name: str = Form(...)):
    """Load demo dataset and run PHI scan."""
    path = get_demo_path(demo_name)
    if not path:
        raise HTTPException(404, f"Demo not found: {demo_name}")
    df = pd.read_csv(path)
    scanner = PHIScanner()
    is_safe, violations = scanner.scan_dataset(df)
    return {
        "is_safe": is_safe,
        "violations": violations,
        "rows": len(df),
        "columns": len(df.columns),
        "data": df_to_json(df),
    }

# ---------------------------------------------------------------------------
# API: Synthetic data
# ---------------------------------------------------------------------------

@app.post("/api/synthetic/generate")
async def synthetic_generate(
    file: UploadFile = File(None),
    demo_name: Optional[str] = Form(None),
    n_samples: int = Form(None),
):
    """Generate synthetic twin. Provide either file or demo_name."""
    if file and file.filename:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    elif demo_name:
        path = get_demo_path(demo_name)
        if not path:
            raise HTTPException(404, f"Demo not found: {demo_name}")
        df = pd.read_csv(path)
    else:
        raise HTTPException(400, "Provide file or demo_name")
    if n_samples is None:
        n_samples = len(df)
    generator = SyntheticDataGenerator()
    generator.fit(df)
    synthetic_df = generator.generate(n_samples=n_samples)
    validation = generator.validate_privacy(df, synthetic_df)
    return {
        "synthetic_data": df_to_json(synthetic_df),
        "validation": _make_serializable(validation),
        "original_rows": len(df),
        "synthetic_rows": len(synthetic_df),
    }

# ---------------------------------------------------------------------------
# API: Quality analysis
# ---------------------------------------------------------------------------

@app.post("/api/analysis/quality")
async def analysis_quality(body: Dict[str, Any]):
    """Run data quality analysis on provided data (list of row objects)."""
    rows = body.get("data")
    if not rows:
        raise HTTPException(400, "Missing 'data' (array of row objects)")
    df = json_to_df(rows)
    analyzer = DataQualityAnalyzer(df)
    issues = analyzer.run_full_analysis()
    summary = analyzer.get_summary()
    return {
        "issues": _make_serializable(issues),
        "summary": _make_serializable(summary),
    }

# ---------------------------------------------------------------------------
# API: Bias analysis
# ---------------------------------------------------------------------------

@app.post("/api/analysis/bias")
async def analysis_bias(body: Dict[str, Any]):
    """Run bias analysis. Optional target_col; auto-detect if not provided."""
    rows = body.get("data")
    if not rows:
        raise HTTPException(400, "Missing 'data'")
    df = json_to_df(rows)
    target_col = body.get("target_col")
    if not target_col:
        possible = [c for c in df.columns if any(t in c.lower() for t in ["disease", "diabetes", "outcome", "target", "recovered"])]
        target_col = possible[0] if possible else df.columns[-1]
    detector = BiasDetector(df, target_col=target_col)
    issues = detector.run_full_analysis()
    summary = detector.get_summary()
    return {
        "issues": _make_serializable(issues),
        "summary": _make_serializable(summary),
        "target_col": target_col,
    }

# ---------------------------------------------------------------------------
# API: Implement changes (apply selected recommendations to synthetic data)
# ---------------------------------------------------------------------------

class ImplementRequest(BaseModel):
    data: List[Dict[str, Any]]
    selected_keys: List[str]
    quality_issues: Dict[str, Any]
    recommendations: List[Dict[str, Any]]  # full list with 'key', 'type', 'column', 'method', etc.


class AskAIRequest(BaseModel):
    question: str
    user_context: Dict[str, Any]
    quality_summary: Dict[str, Any]
    bias_summary: Dict[str, Any]


class FairnessSpecialistRequest(BaseModel):
    """One demographic attribute; mirrors Streamlit Tab 2 Fairness Specialist call."""

    attribute: str
    distribution: Dict[str, float]
    issues: List[str]
    user_context: Dict[str, Any]
    total_samples: int


@app.post("/api/implement")
async def implement_changes(req: ImplementRequest):
    """Apply selected recommendations to the dataset. Returns modified data and before/after summary."""
    import numpy as np
    modified_df = json_to_df(req.data).copy()
    recommendations_by_key = {r["key"]: r for r in req.recommendations}
    quality_issues = req.quality_issues
    applied = []

    def _summarize(df: pd.DataFrame) -> Dict[str, Any]:
        missing_per_column = {}
        for c in df.columns:
            try:
                n = int(pd.isna(df[c]).sum())
                if n > 0:
                    missing_per_column[c] = n
            except Exception:
                pass
        dup_count = int(df.duplicated().sum()) if len(df) else 0
        return {
            "total_rows": len(df),
            "duplicate_count": dup_count,
            "missing_per_column": missing_per_column,
            "columns": list(df.columns),
        }

    before_summary = _summarize(modified_df)

    for key in req.selected_keys:
        rec = recommendations_by_key.get(key)
        if not rec:
            continue
        try:
            if rec.get("type") == "missing_value":
                col = rec.get("column")
                if not col:
                    continue
                val = rec.get("value")
                if "Median" in rec.get("method", ""):
                    if val is None:
                        val = modified_df[col].median()
                    modified_df[col] = modified_df[col].fillna(val)
                elif "Mean" in rec.get("method", ""):
                    if val is None:
                        val = modified_df[col].mean()
                    modified_df[col] = modified_df[col].fillna(val)
                elif "Mode" in rec.get("method", ""):
                    if val is None:
                        mode_vals = modified_df[col].mode()
                        val = mode_vals[0] if len(mode_vals) else None
                    if val is not None:
                        modified_df[col] = modified_df[col].fillna(val)
                elif "Remove" in rec.get("method", ""):
                    modified_df = modified_df.dropna(subset=[col])
                applied.append(f"Missing values: {col}")

            elif rec.get("type") == "duplicate":
                before = len(modified_df)
                modified_df = modified_df.drop_duplicates()
                applied.append(f"Removed {before - len(modified_df)} duplicates")

            elif rec.get("type") == "outlier" and rec.get("column") and quality_issues.get("outliers", {}).get(rec["column"]):
                bounds = quality_issues["outliers"][rec["column"]]["bounds"]
                modified_df[rec["column"]] = modified_df[rec["column"]].clip(
                    lower=bounds["lower"], upper=bounds["upper"]
                )
                applied.append(f"Outliers clipped: {rec['column']}")

            elif rec.get("type") == "bias_normalization" and rec.get("column"):
                detector = BiasDetector(modified_df)
                modified_df[rec["column"]] = detector._normalize_sex_gender_values(modified_df[rec["column"]])
                applied.append(f"Normalized: {rec['column']}")

            elif rec.get("type") == "bias_mitigation" and rec.get("column"):
                col_name = rec["column"]
                method = rec.get("method", "")
                if "SMOTE" in method:
                    try:
                        from imblearn.over_sampling import SMOTE
                        X = modified_df.drop(columns=[col_name])
                        y = modified_df[col_name]
                        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            X_numeric = X[numeric_cols].values
                            smote = SMOTE(random_state=42)
                            X_res, y_res = smote.fit_resample(X_numeric, y)
                            resampled = pd.DataFrame(X_res, columns=numeric_cols)
                            resampled[col_name] = y_res
                            cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
                            for c in cat_cols:
                                resampled[c] = None
                            for i in range(min(len(modified_df), len(resampled))):
                                for c in cat_cols:
                                    resampled.iloc[i, resampled.columns.get_loc(c)] = modified_df.iloc[i][c]
                            if len(resampled) > len(modified_df):
                                minority = y_res.iloc[len(modified_df)] if hasattr(y_res, 'iloc') else y_res[len(modified_df)]
                                minority_df = modified_df[modified_df[col_name] == minority]
                                for i in range(len(modified_df), len(resampled)):
                                    row = minority_df.sample(n=1, random_state=42 + i).iloc[0]
                                    for c in cat_cols:
                                        resampled.iloc[i, resampled.columns.get_loc(c)] = row[c]
                            modified_df = resampled
                            applied.append(f"SMOTE on {col_name}")
                        else:
                            vc = modified_df[col_name].value_counts()
                            minority_group = vc.idxmin()
                            majority_count = vc.max()
                            minority_df = modified_df[modified_df[col_name] == minority_group]
                            need = majority_count - len(minority_df)
                            oversampled = minority_df.sample(n=int(need), replace=True, random_state=42)
                            modified_df = pd.concat([modified_df, oversampled]).reset_index(drop=True)
                            applied.append(f"Oversampling on {col_name}")
                    except Exception as e:
                        applied.append(f"SMOTE failed: {e}")
                elif "undersampling" in method.lower():
                    vc = modified_df[col_name].value_counts()
                    minority_group = vc.idxmin()
                    minority_count = vc[minority_group]
                    minority_df = modified_df[modified_df[col_name] == minority_group]
                    majority_df = modified_df[modified_df[col_name] != minority_group]
                    majority_sampled = majority_df.sample(n=int(minority_count), random_state=42)
                    modified_df = pd.concat([minority_df, majority_sampled]).reset_index(drop=True)
                    applied.append(f"Undersampling on {col_name}")
        except Exception as e:
            applied.append(f"Error applying {key}: {e}")

    after_summary = _summarize(modified_df)

    return {
        "data": df_to_json(modified_df),
        "applied": applied,
        "before_summary": _make_serializable(before_summary),
        "after_summary": _make_serializable(after_summary),
    }


@app.post("/api/ask-ai")
async def ask_ai(body: AskAIRequest):
    """
    Lightweight chat endpoint similar to Streamlit's 'Ask AI' tab.
    Uses quality/bias summaries and user context to answer a free-text question.
    """
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    ctx = body.user_context or {}
    qsum = body.quality_summary or {}
    bsum = body.bias_summary or {}

    context_summary = f"""
User's project:
- Model: {ctx.get('model_type', 'Unknown')}
- Timeline: {ctx.get('timeline_days', 'Unknown')} days
- Use case: {ctx.get('use_case', 'Unknown')}
- Location: {ctx.get('location', 'Unknown')}

Dataset analysis:
- Total records: {qsum.get('total_records', 'Unknown')}
- Quality issues: {qsum.get('missing_value_columns', 0)} missing value columns, {qsum.get('duplicate_records', 0)} duplicates
- Bias status: {bsum.get('status', 'Unknown')}
"""

    try:
        response = ollama.generate(
            model="llama3.2:3b",
            prompt=f"""
You are a helpful medical data science advisor.

CONTEXT:
{context_summary}

USER QUESTION:
{question}

Provide a clear, specific answer in 2-4 sentences.
Use exact numbers when referencing the data.
Be helpful and educational.
""",
            options={"temperature": 0.3},
        )
    except Exception as e:
        err_msg = str(e)
        if "10061" in err_msg or "refused" in err_msg.lower() or "connection" in err_msg.lower():
            return {"answer": None, "error": "Ollama is not running. Start Ollama (run 'ollama serve' or open the Ollama app), then run: ollama pull llama3.2:3b"}
        return {"answer": None, "error": f"AI chat failed: {err_msg}"}

    return {"answer": response.get("response", "")}

# ---------------------------------------------------------------------------
# API: Build recommendations list (for Implement tab)
# ---------------------------------------------------------------------------

@app.post("/api/recommendations")
async def get_recommendations(body: Dict[str, Any]):
    """Build full list of recommendations from quality_issues and bias_issues."""
    quality_issues = body.get("quality_issues", {})
    bias_issues = body.get("bias_issues", {})
    all_recs = []

    for col, info in quality_issues.get("missing_values", {}).items():
        for r in info.get("recommendations", []):
            if "code" in r and r.get("priority") not in ["❌ NOT RECOMMENDED", "❌ CRITICAL"]:
                all_recs.append({
                    "type": "missing_value", "column": col, "method": r["method"],
                    "priority": r["priority"], "reason": r["reason"], "code": r["code"],
                    "value": r.get("value"), "impact": r.get("impact", ""),
                    "key": f"missing_{col}_{r['method'].replace(' ', '_').lower()[:20]}",
                })

    dup = quality_issues.get("duplicates", {})
    if dup.get("count", 0) > 0 and dup.get("recommendation"):
        r = dup["recommendation"]
        all_recs.append({
            "type": "duplicate", "column": None, "method": r["method"],
            "priority": r["priority"], "reason": r["reason"], "code": r["code"],
            "value": None, "impact": r.get("impact", ""), "key": "duplicate_remove",
        })

    for col, info in quality_issues.get("outliers", {}).items():
        for r in info.get("recommendations", []):
            if "code" in r:
                all_recs.append({
                    "type": "outlier", "column": col, "method": r["method"],
                    "priority": r["priority"], "reason": r["reason"], "code": r["code"],
                    "value": None, "impact": r.get("impact", ""),
                    "key": f"outlier_{col}_{r['method'].replace(' ', '_').lower()[:15]}",
                })

    for col, info in (bias_issues or {}).items():
        if "sex" in col.lower() or "gender" in col.lower():
            dist = info.get("distribution", {})
            keys = list(dist.keys())
            if len(keys) > 2 or any(k not in ["Male", "Female"] for k in keys):
                all_recs.append({
                    "type": "bias_normalization", "column": col, "method": "Normalize Sex/Gender",
                    "priority": "⭐ RECOMMENDED", "reason": "Normalize to Male/Female", "code": "",
                    "value": None, "impact": "Consistent demographics", "key": f"bias_norm_{col}",
                })
        if info.get("bias_detected"):
            dist_vals = list(info.get("distribution", {}).values())
            if len(dist_vals) >= 2:
                gap = abs(max(dist_vals) - min(dist_vals))
                if gap > 5:
                    all_recs.append({
                        "type": "bias_mitigation", "column": col, "method": "SMOTE",
                        "priority": "⭐ RECOMMENDED", "reason": f"Balance (gap {gap:.1f}%)",
                        "code": "from imblearn.over_sampling import SMOTE\nsmote = SMOTE(random_state=42)\nX_balanced, y_balanced = smote.fit_resample(X, y)",
                        "value": None, "impact": f"Reduce gap from {gap:.1f}%", "key": f"bias_smote_{col}",
                    })

    return {"recommendations": all_recs}

# ---------------------------------------------------------------------------
# API: Deployment plan (Deployment Strategist - same as Streamlit)
# ---------------------------------------------------------------------------

@app.post("/api/deployment-plan")
async def get_deployment_plan(body: Dict[str, Any]):
    """Generate week-by-week deployment plan from quality, bias, and user context."""
    quality_issues = body.get("quality_issues", {})
    bias_summary = body.get("bias_summary", {})
    user_context = body.get("user_context", {})
    quality_summary_dict = {
        "missing_values": len(quality_issues.get("missing_values", {})),
        "duplicates": quality_issues.get("duplicates", {}).get("count", 0) if quality_issues.get("duplicates") else 0,
        "outliers": len(quality_issues.get("outliers", {})),
    }
    bias_summary_dict = {
        "biased_attributes": bias_summary.get("attributes_with_bias", 0),
        "status": bias_summary.get("status", "UNKNOWN"),
    }
    fairness_rec_dict = body.get("fairness_recommendations") or {
        "exact_samples_needed": {},
        "timeline_months": 6,
        "immediate_method": "SMOTE",
    }
    try:
        strategist = DeploymentStrategist()
        plan = strategist.create_plan(
            quality_summary_dict,
            bias_summary_dict,
            fairness_rec_dict,
            user_context,
        )
        out = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
        return _make_serializable(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deployment plan failed: {str(e)}")


@app.post("/api/fairness/specialist")
async def fairness_specialist_analyze(req: FairnessSpecialistRequest):
    """
    Run Fairness Specialist (Ollama + schema) for one biased attribute.
    Same inputs as Streamlit app_complete.py bias tab.
    """
    if not req.issues:
        raise HTTPException(status_code=400, detail="issues required when bias is detected")
    dist = {str(k): float(v) for k, v in req.distribution.items()}
    if not dist:
        raise HTTPException(status_code=400, detail="distribution cannot be empty")

    first_issue = req.issues[0]
    minority_group = first_issue.split(":")[0].strip() if first_issue else "underrepresented"
    bias_data = {
        "type": f"{req.attribute}_imbalance",
        "distribution": dist,
        "minority_group": minority_group,
    }
    dist_values = list(dist.values())
    majority_pct = max(dist_values)
    minority_pct = min(dist_values)
    total = max(0, int(req.total_samples))
    majority_count = int(total * majority_pct / 100.0)
    minority_count = int(total * minority_pct / 100.0)
    samples_needed = max(0, majority_count - minority_count)
    stats = {
        "total_samples": total,
        "samples_needed": samples_needed,
        "imbalance_pct": abs(majority_pct - 50.0),
    }
    try:
        specialist = _get_fairness_specialist()
        rec = specialist.analyze_bias(bias_data, req.user_context, stats)
        out = rec.model_dump() if hasattr(rec, "model_dump") else rec.dict()
        return _make_serializable(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fairness specialist failed: {str(e)}")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/demos")
def list_demos():
    available = []
    for name, path in DEMO_FILES.items():
        full = ROOT / path
        if full.exists():
            available.append(name)
    return {"demos": available}
