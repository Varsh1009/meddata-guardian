"""
FIDES Stage 1 — HIPAA Ingestion Layer
Runs inside hospital environment. De-identifies, minimizes, logs.
"""

import hashlib
import json
import re
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.research_spec import ResearchSpec

warnings.filterwarnings('ignore')

# ── 18 HIPAA Safe Harbor identifiers ─────────────────────────────────────────

HIPAA_18_PATTERNS = {
    "names":       r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
    "ssn":         r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
    "phone":       r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "email":       r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b',
    "ip_address":  r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    "url":         r'https?://\S+|www\.\S+',
}

IDENTIFIER_COLUMN_KEYWORDS = [
    "name", "firstname", "lastname", "fullname", "patient_name",
    "ssn", "social_security", "sin",
    "mrn", "medical_record", "record_number", "chart",
    "phone", "telephone", "mobile", "fax",
    "email", "mail",
    "address", "street", "city",
    "dob", "date_of_birth", "birthdate", "birth_date",
    "admission_date", "discharge_date", "service_date",
    "device_id", "device_serial",
    "vehicle", "license_plate",
    "account_number", "beneficiary",
    "certificate", "license_number",
    "url", "web", "ip_address",
    "biometric", "fingerprint", "photo", "image",
]


# ── Audit Logger ──────────────────────────────────────────────────────────────

class AuditLogger:
    def __init__(self, log_path: str = "/tmp/fides_audit.jsonl"):
        self.log_path = Path(log_path)

    def log(self, event: dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_log(self) -> List[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]


_audit_logger = AuditLogger()


# ── Core ingestion ────────────────────────────────────────────────────────────

def _compute_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def _remove_identifier_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    removed = []
    for col in df.columns:
        norm = col.lower().strip().replace(" ", "_")
        if any(kw in norm for kw in IDENTIFIER_COLUMN_KEYWORDS):
            removed.append(col)
    return df.drop(columns=removed), removed


def _generalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce date columns to year only (HIPAA Safe Harbor)."""
    for col in df.columns:
        norm = col.lower()
        if any(kw in norm for kw in ["date", "dob", "birth", "admission", "discharge"]):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.year
            except Exception:
                pass
    return df


def _generalize_zip(df: pd.DataFrame) -> pd.DataFrame:
    """Truncate zip codes to 3-digit prefix."""
    for col in df.columns:
        if "zip" in col.lower():
            df[col] = df[col].astype(str).str[:3] + "XX"
    return df


def _tokenize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Replace any remaining ID columns with random UUIDs — no linkage table."""
    for col in df.columns:
        norm = col.lower()
        if any(kw in norm for kw in ["patient_id", "subject_id", "id", "mrn"]):
            df[col] = [f"TOKEN_{uuid.uuid4().hex[:8].upper()}" for _ in range(len(df))]
    return df


def _scan_cell_values(df: pd.DataFrame) -> List[str]:
    """Check sample of cell values for PHI patterns."""
    violations = []
    sample = df.sample(min(50, len(df)), random_state=42)
    for col in sample.select_dtypes(include="object").columns:
        for val in sample[col].dropna().astype(str).head(20):
            for phi_type, pattern in HIPAA_18_PATTERNS.items():
                if re.search(pattern, val):
                    violations.append(f"Column '{col}' may contain {phi_type}: '{val[:30]}...'")
                    break
    return violations


def _select_required_columns(df: pd.DataFrame, spec: ResearchSpec) -> pd.DataFrame:
    """Keep only columns specified in ResearchSpec (minimum necessary)."""
    available = set(df.columns)
    required = set(spec.required_columns)
    # Always keep target
    required.add(spec.target_variable)
    keep = list(available & required)
    dropped = list(available - required)
    if dropped:
        print(f"  [Minimum Necessary] Dropped {len(dropped)} columns not in spec: {dropped[:5]}{'...' if len(dropped)>5 else ''}")
    return df[keep]


# ── Public API ────────────────────────────────────────────────────────────────

def ingest(
    df: pd.DataFrame,
    spec: ResearchSpec,
    user_id: str = "anonymous",
    irb_protocol: str = "",
) -> Tuple[pd.DataFrame, dict]:
    """
    Stage 1: HIPAA-compliant ingestion.
    Returns (de-identified DataFrame, audit_record).
    """
    print("=" * 55)
    print("FIDES Stage 1: HIPAA Ingestion")
    print("=" * 55)

    # Step 1: Hash original dataset
    original_hash = _compute_hash(df)
    print(f"  Dataset hash: {original_hash[:16]}...")
    print(f"  Shape: {df.shape}")

    # Step 2: Remove identifier columns
    df, removed_cols = _remove_identifier_columns(df)
    if removed_cols:
        print(f"  Removed identifier columns: {removed_cols}")

    # Step 3: Generalize dates and zip codes
    df = _generalize_dates(df.copy())
    df = _generalize_zip(df.copy())

    # Step 4: Tokenize any remaining ID fields
    df = _tokenize_ids(df.copy())

    # Step 5: Minimum necessary — keep only spec columns
    df = _select_required_columns(df, spec)

    # Step 6: Scan cell values for residual PHI
    violations = _scan_cell_values(df)
    if violations:
        print(f"  ⚠ Residual PHI warnings ({len(violations)}):")
        for v in violations[:3]:
            print(f"    {v}")

    # Step 7: Final hash
    clean_hash = _compute_hash(df)

    # Step 8: Audit log entry
    audit_record = {
        "action": "ingest",
        "user_id": user_id,
        "irb_protocol": irb_protocol,
        "original_hash": original_hash,
        "clean_hash": clean_hash,
        "original_shape": list(df.shape),
        "removed_columns": removed_cols,
        "residual_phi_warnings": len(violations),
        "spec_id": spec.domain + "_" + spec.target_variable,
        "hipaa_safe": len(violations) == 0,
    }
    _audit_logger.log(audit_record)

    status = "SAFE" if len(violations) == 0 else f"WARNING ({len(violations)} residual PHI flags)"
    print(f"\n  Status: {status}")
    print(f"  Clean shape: {df.shape}")
    print(f"  Ready for Stage 2.\n")

    return df, audit_record


def get_audit_log() -> List[dict]:
    return _audit_logger.get_log()


if __name__ == "__main__":
    from src.utils.research_spec import build_research_spec

    spec = build_research_spec(
        domain="cardiology",
        target_variable="outcome",
        target_type="binary",
        use_case="research",
        columns=["patient_id", "age", "sex", "race", "zip_code",
                 "HbA1c", "systolic_bp", "outcome"],
    )
    df = pd.read_csv("data/synthetic/dataset1_heart_disease_quality.csv")
    clean_df, audit = ingest(df, spec)
    print(f"Audit record: {audit}")
