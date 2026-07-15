"""
config.py — Centralized configuration for the XGBoost Survival Pipeline.

All column classifications, ordinal mappings, hyperparameters, and time horizons
are defined here to keep the rest of the pipeline clean and DRY.
"""
from typing import Dict, List
import os
import sqlite3
import hashlib
import random
from contextlib import contextmanager
# ============================================================================
#  PATH CONFIGURATIONS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODEL_DIR = os.path.join(BASE_DIR, "pipeline")

OLTP_PATH = os.path.join(DATA_DIR, "oltp_hr.db")
OLAP_PATH = os.path.join(DATA_DIR, "olap_warehouse.db")

# ============================================================================
#  DATABASE CONNECTION MANAGER
# ============================================================================
@contextmanager
def get_conn(db_path: str):
    """Context manager for SQLite database connections."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

# ============================================================================
#  COLUMN CLASSIFICATIONS
# ============================================================================

PERFORMANCE_SCORE_LABELS = {1: "Best", 2: "Good", 3: "Average", 4: "Bad"}
HIGH_PERFORMANCE = {1, 2}
LOW_PERFORMANCE = {3, 4}

def generate_performance_score(employee_id: str, job_satisfaction: int, 
                                 job_involvement: int, performance_rating: int) -> int:
    """
    Deterministic synthetic score correlated with existing sentiment fields
    so it looks plausible, not random. Same employee always gets same score
    on re-run (hash-seeded), but weighted toward existing satisfaction signal.
    """
    seed = int(hashlib.md5(employee_id.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    # base signal: average of existing ordinal sentiment fields (already 1-4 scale, and 1-3 for performance rating)
    base = (job_satisfaction + job_involvement + performance_rating) / 3
    noise = rng.uniform(-0.6, 0.6)
    score = round(base + noise)
    return max(1, min(4, score))

# Survival target columns (used by the model, excluded from features)
DURATION_COL: str = "YearsAtCompany"
EVENT_COL: str = "Attrition"

# Ordinal categorical columns — have a meaningful rank order
ORDINAL_COLS: List[str] = [
    "BusinessTravel",
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]

# Ordinal encoding mappings (text label -> numeric rank)
ORDINAL_MAPPINGS: Dict[str, Dict[str, int]] = {
    "BusinessTravel": {"Non-Travel": 1, "Travel_Rarely": 2, "Travel_Frequently": 3},
    "Education": {
        "Below College": 1,
        "College": 2,
        "Bachelor": 3,
        "Master": 4,
        "Doctor": 5,
    },
    "EnvironmentSatisfaction": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
    "JobInvolvement": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
    "JobLevel": {
        "Entry Level": 1,
        "Junior Level": 2,
        "Mid Level": 3,
        "Senior Level": 4,
        "Executive Level": 5,
    },
    "JobSatisfaction": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
    "PerformanceRating": {"Good": 1, "Excellent": 2, "Outstanding": 3},
    "RelationshipSatisfaction": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
    "WorkLifeBalance": {"Bad": 1, "Low": 2, "Good": 3, "Better": 4, "Best": 5},
}

# Low-cardinality nominal columns — one-hot encoded
ONE_HOT_COLS: List[str] = ["Gender", "MaritalStatus", "OverTime", "Department", "Project"]

# High-cardinality nominal columns — Leave-One-Out target encoded
LOO_COLS: List[str] = ["JobRole", "EducationField"]

# Columns to skip during numeric casting (remain as text until encoding)
SKIP_NUMERIC_CAST: List[str] = [
    "Attrition",
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "Project",
    "DateOfLeaving"
]

# ============================================================================
#  XGBOOST HYPERPARAMETERS
# ============================================================================

XGBOOST_PARAMS: Dict = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "gamma": 1.0,
    "random_state": 42,
    "verbosity": 0,
}

# Early stopping rounds during training
EARLY_STOPPING_ROUNDS: int = 30

# ============================================================================
#  TIME HORIZONS (in years)
# ============================================================================

# Maps human-readable labels to fractional years
TIME_HORIZONS: Dict[str, float] = {
    "1-Month": 1 / 12,
    "3-Month": 3 / 12,
    "6-Month": 6 / 12,
    "12-Month": 1.0,
}

# ============================================================================
#  TRAIN / TEST SPLIT
# ============================================================================

TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ============================================================================
#  REPORTING
# ============================================================================

# Risk threshold (probability) above which an employee is flagged as high-risk
HIGH_RISK_THRESHOLD: float = 0.30  # 30% at any horizon

# Maximum number of employees to display in the risk profile table
MAX_DISPLAY_EMPLOYEES: int = 20
