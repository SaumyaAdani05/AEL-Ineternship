"""
server.py — FastAPI analytical backend server.

Exposes REST APIs for:
  - Paginated employee risk listings
  - Historical risk progression analysis from SCD Type 2 logs
  - Real-time What-If model inference (XGBoost)
  - Simulating HR actions (promotions, overtime changes)
"""
import os
import sys

# Add project root to sys.path so 'pipeline' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import sqlite3
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal, List
from datetime import datetime
from neo4j import GraphDatabase, exceptions

# Suppress noisy sklearn version mismatch warnings when loading pickled models
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Reconfigure stdout to support unicode symbols
sys.stdout.reconfigure(encoding='utf-8')

# Paths and configuration
from pipeline.config import OLAP_PATH, MODEL_DIR


# In-memory pipeline cache
PIPELINE = {}

def load_pipeline():
    """
    Load model checkpoints and encoders from the pipeline directory.
    """
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    loo_path = os.path.join(MODEL_DIR, "loo_encoder.pkl")
    cph_path = os.path.join(MODEL_DIR, "model_cph.pkl")
    xgb_path = os.path.join(MODEL_DIR, "model_xgb.json")
    base_path = os.path.join(MODEL_DIR, "baseline_survival.pkl")
    
    if not (os.path.exists(scaler_path) and os.path.exists(xgb_path)):
        raise RuntimeError("Model checkpoints not found. Run pipeline/production_ml.py first.")
        
    with open(scaler_path, "rb") as f:
        PIPELINE["scaler"] = pickle.load(f)
    with open(loo_path, "rb") as f:
        PIPELINE["loo_encoder"] = pickle.load(f)
    with open(cph_path, "rb") as f:
        PIPELINE["cph"] = pickle.load(f)
    with open(base_path, "rb") as f:
        PIPELINE["baseline_survival"] = pickle.load(f)
        
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model(xgb_path)
    PIPELINE["model_xgb"] = model
    
    # Store training feature names
    PIPELINE["feature_names"] = model.feature_names
    print("[+] Model checkpoints successfully loaded into FastAPI cache.")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    """Modern lifespan handler replacing deprecated on_event('startup')."""
    load_pipeline()
    
    # Initialize Neo4j driver
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        driver.verify_connectivity()
        app.state.neo4j_driver = driver
    except Exception as e:
        print(f"[!] Warning: Could not connect to Neo4j. Graph features will run in mock mode. Error: {e}")
        app.state.neo4j_driver = None
        
    yield
    
    if getattr(app.state, "neo4j_driver", None):
        app.state.neo4j_driver.close()

app = FastAPI(title="MNC HR Attrition Risk REST Service", lifespan=lifespan)

# ============================================================================
#  DATA MODELLING & SCHEMAS
# ============================================================================

class WhatIfRequest(BaseModel):
    EmployeeId: str
    MonthlyIncome: Optional[float] = Field(None, ge=0)
    OverTime: Optional[Literal["Yes", "No"]] = None
    YearsWithCurrManager: Optional[int] = Field(None, ge=0)
    JobRole: Optional[str] = None
    JobLevel: Optional[Literal["Entry Level", "Junior Level", "Mid Level", "Senior Level", "Executive Level"]] = None

class ActionParams(BaseModel):
    JobRole: Optional[str] = None
    JobLevel: Optional[Literal["Entry Level", "Junior Level", "Mid Level", "Senior Level", "Executive Level"]] = None
    MonthlyIncome: Optional[float] = Field(None, ge=0)
    OverTime: Optional[Literal["Yes", "No"]] = None
    YearsWithCurrManager: Optional[int] = Field(None, ge=0)
    PercentSalaryHike: Optional[float] = Field(None, ge=0)

class ActionRequest(BaseModel):
    action: Literal["promote", "overtime", "manager_change", "salary_hike"]
    EmployeeId: str
    params: ActionParams

# ============================================================================
#  API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    dash_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(dash_path, "r", encoding="utf-8") as f:
        return f.read()

import json
@app.get("/api/model/metadata")
def get_model_metadata():
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {"error": "Metadata not found."}

@app.get("/api/employees")
def list_employees(
    page: int = 1,
    limit: int = 10,
    search: str = "",
    status: str = "all",
    sort_by: str = "General_Risk_Score",
    sort_dir: str = "desc",
    department: str = "",
):
    conn = sqlite3.connect(OLAP_PATH)
    
    # Get active cohort with their current computed risk scores
    query = """
    SELECT 
        v.*,
        r.Prob_Leave_1M, r.Prob_Leave_3M, r.Prob_Leave_6M, r.Prob_Leave_12M, r.General_Risk_Score,
        r.Contrib_Identity, r.Contrib_Environment, r.Contrib_Compensation, r.Contrib_Sentiment, r.Contrib_Tenure
    FROM v_ml_features v
    LEFT JOIN flight_risk_scores r ON v.EmployeeId = r.EmployeeId
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Unique departments for filter dropdown
    departments = sorted(df["Department"].dropna().unique().tolist())
    
    # Apply search filter
    if search:
        df = df[df["EmployeeId"].str.contains(search, case=False)]
    
    # Apply department filter
    if department:
        df = df[df["Department"] == department]
        
    # Apply status filter
    # High risk = exceeds 30% attrition threshold at any horizon
    is_high_risk = (df["Prob_Leave_1M"] >= 0.30) | (df["Prob_Leave_3M"] >= 0.30) | (df["Prob_Leave_6M"] >= 0.30) | (df["Prob_Leave_12M"] >= 0.30)
    
    total_active = len(is_high_risk)
    high_risk_count = int(is_high_risk.sum())
    
    # Risk distribution buckets for donut chart
    risk_distribution = {
        "critical": int((df["General_Risk_Score"] >= 80).sum()),
        "high": int(((df["General_Risk_Score"] >= 50) & (df["General_Risk_Score"] < 80)).sum()),
        "medium": int(((df["General_Risk_Score"] >= 20) & (df["General_Risk_Score"] < 50)).sum()),
        "low": int((df["General_Risk_Score"] < 20).sum()),
    }
    
    if status == "high":
        df = df[is_high_risk]
    elif status == "normal":
        df = df[~is_high_risk]
        
    total_filtered = len(df)
    
    # Sort
    valid_sort_cols = {
        "General_Risk_Score", "MonthlyIncome", "YearsAtCompany",
        "Prob_Leave_1M", "Prob_Leave_12M", "Age", "EmployeeId"
    }
    if sort_by in valid_sort_cols:
        df = df.sort_values(sort_by, ascending=(sort_dir == "asc"))
    else:
        df = df.sort_values("General_Risk_Score", ascending=False)
    
    # Paginate
    start = (page - 1) * limit
    end = start + limit
    df_slice = df.iloc[start:end]
    
    # Format probabilities as percentages for display
    employees_list = []
    for _, row in df_slice.iterrows():
        employees_list.append({
            "EmployeeId": row["EmployeeId"],
            "Age": int(row["Age"]),
            "Gender": row["Gender"],
            "Department": row["Department"],
            "JobRole": row["JobRole"],
            "JobLevel": row["JobLevel"],
            "MonthlyIncome": float(row["MonthlyIncome"]),
            "OverTime": row["OverTime"],
            "YearsAtCompany": int(row["YearsAtCompany"]),
            "YearsWithCurrManager": int(row["YearsWithCurrManager"]),
            "Prob_1M": f"{row['Prob_Leave_1M'] * 100:.1f}%",
            "Prob_3M": f"{row['Prob_Leave_3M'] * 100:.1f}%",
            "Prob_6M": f"{row['Prob_Leave_6M'] * 100:.1f}%",
            "Prob_12M": f"{row['Prob_Leave_12M'] * 100:.1f}%",
            "General_Risk_Score": round(float(row["General_Risk_Score"]), 1),
            "is_high_risk": bool(any(row[p] >= 0.30 for p in ["Prob_Leave_1M", "Prob_Leave_3M", "Prob_Leave_6M", "Prob_Leave_12M"])),
            "contributions": {
                "identity": round(float(row["Contrib_Identity"]), 1),
                "environment": round(float(row["Contrib_Environment"]), 1),
                "compensation": round(float(row["Contrib_Compensation"]), 1),
                "sentiment": round(float(row["Contrib_Sentiment"]), 1),
                "tenure": round(float(row["Contrib_Tenure"]), 1)
            }
        })
        
    return {
        "employees": employees_list,
        "total": total_filtered,
        "high_risk_count": high_risk_count,
        "total_active": total_active,
        "departments": departments,
        "risk_distribution": risk_distribution,
    }

@app.get("/api/dashboard/stats")
def get_dashboard_stats():
    conn = sqlite3.connect(OLAP_PATH)
    
    query = """
    SELECT 
        v.Department,
        r.General_Risk_Score,
        r.Prob_Leave_1M, r.Prob_Leave_3M, r.Prob_Leave_6M, r.Prob_Leave_12M,
        r.Contrib_Identity, r.Contrib_Environment, r.Contrib_Compensation, r.Contrib_Sentiment, r.Contrib_Tenure
    FROM v_ml_features v
    JOIN flight_risk_scores r ON v.EmployeeId = r.EmployeeId
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        return {}
        
    total_active = len(df)
    is_high_risk = (df["Prob_Leave_1M"] >= 0.30) | (df["Prob_Leave_3M"] >= 0.30) | (df["Prob_Leave_6M"] >= 0.30) | (df["Prob_Leave_12M"] >= 0.30)
    high_risk_count = int(is_high_risk.sum())
    
    avg_risk_score = float(df["General_Risk_Score"].mean())
    
    # Risk by department (sort by highest risk)
    dept_risk_series = df.groupby("Department")["General_Risk_Score"].mean().sort_values(ascending=False)
    dept_risk = {k: float(v) for k, v in dept_risk_series.items()}
    

    
    return {
        "total_active": total_active,
        "high_risk_count": high_risk_count,
        "avg_risk_score": avg_risk_score,
        "department_risk": dept_risk
    }


def encode_single_row(row_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a single employee row into the exact scaled, encoded feature space
    expected by the trained models.
    """
    from pipeline import config
    from pipeline.data_pipeline import preprocess_data
    
    if 'DateOfLeaving' in row_df.columns:
        row_df = row_df.drop(columns=['DateOfLeaving'])
    
    # 1. Preprocess (mappings, types, zero-duration fix)
    df_clean = preprocess_data(row_df)
    # Reset to integer index so .loc[0, col] works regardless of original index
    df_clean = df_clean.reset_index(drop=True)
    
    # 2. Re-create nominal One-Hot dummies
    # We must match the exact feature names list in PIPELINE["feature_names"]
    # Build a base matrix containing 1 row of all zeros
    cols = PIPELINE["feature_names"]
    x_encoded = pd.DataFrame(0, index=[0], columns=cols)
    
    # Populate numeric and ordinal fields (skip text columns — they're handled by OHE/LOO below)
    text_cols = set(config.ONE_HOT_COLS + config.LOO_COLS)
    for col in df_clean.columns:
        if col in cols and col not in text_cols:
            # Skip any remaining non-numeric values
            val = df_clean.loc[0, col]
            if isinstance(val, str):
                continue
            # Scale continuous columns
            if col in PIPELINE["scaler"].feature_names_in_:
                col_idx = list(PIPELINE["scaler"].feature_names_in_).index(col)
                min_val = PIPELINE["scaler"].data_min_[col_idx]
                max_val = PIPELINE["scaler"].data_max_[col_idx]
                scaled_val = (val - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.0
                x_encoded.loc[0, col] = scaled_val
            else:
                x_encoded.loc[0, col] = val
                
    # OHE Columns
    for ohe_col in config.ONE_HOT_COLS:
        if ohe_col in df_clean.columns:
            val = df_clean.loc[0, ohe_col]
            dummy_col = f"{ohe_col}_{val}"
            if dummy_col in cols:
                x_encoded.loc[0, dummy_col] = 1
                
    # LOO Columns (apply saved encoder)
    # The LOO encoder was fit on a differently-shaped DataFrame (post-OHE).
    # For single-row inference, we extract the per-category mean mappings directly.
    # Encoder mapping: dict of {col_name: DataFrame(sum, count)} — mean = sum/count.
    loo_cols = [c for c in config.LOO_COLS if c in df_clean.columns]
    if loo_cols:
        encoder = PIPELINE["loo_encoder"]
        for col in loo_cols:
            if col in cols and hasattr(encoder, 'mapping') and col in encoder.mapping:
                raw_val = df_clean.loc[0, col]
                mapping_df = encoder.mapping[col]  # DataFrame with 'sum' and 'count'
                if raw_val in mapping_df.index:
                    row_data = mapping_df.loc[raw_val]
                    x_encoded.loc[0, col] = float(row_data['sum'] / row_data['count'])
                else:
                    # Unseen category — use overall mean across all categories
                    overall_mean = float(mapping_df['sum'].sum() / mapping_df['count'].sum())
                    x_encoded.loc[0, col] = overall_mean
                
    # Ensure float types to prevent XGBoost type mismatch
    x_encoded = x_encoded.astype(float)
    return x_encoded

def predict_risk_profile(x_encoded: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates time-horizon probabilities for an encoded feature vector.
    """
    model = PIPELINE["model_xgb"]
    baseline_survival = PIPELINE["baseline_survival"]
    
    dmatrix = xgb.DMatrix(x_encoded)
    raw_pred = model.predict(dmatrix, output_margin=True)[0]
    clipped_pred = np.clip(raw_pred, -15, 15)
    risk_multiplier = np.exp(clipped_pred)
    
    horizons = [1, 3, 6, 12]
    probs = {}
    
    for h in horizons:
        idx = (np.abs(baseline_survival.index - h)).argmin()
        s0_t = baseline_survival.iloc[idx]
        probs[h] = float(1.0 - (s0_t ** risk_multiplier))
        
    return {
        "Prob_1M": probs[1],
        "Prob_3M": probs[3],
        "Prob_6M": probs[6],
        "Prob_12M": probs[12],
        "General_Risk_Score": probs[12] * 100
    }

@app.get("/api/employees/{employee_id}/history")
def get_employee_history(employee_id: str):
    conn = sqlite3.connect(OLAP_PATH)
    
    # Query all SCD Type 2 history records for this employee
    query = """
    SELECT * FROM employee_history 
    WHERE EmployeeId = ? 
    ORDER BY valid_from ASC
    """
    df_hist = pd.read_sql(query, conn, params=(employee_id,))
    conn.close()
    
    if df_hist.empty:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} history not found.")
        
    history_points = []
    
    # For each historical row, run our pipeline to see what their risk score WAS
    for idx, row in df_hist.iterrows():
        # Build single row dataframe
        row_df = pd.DataFrame([row.drop(["row_id", "valid_from", "valid_to", "is_active"])])
        
        # Set EmployeeId as index so it doesn't get coerced to NaN
        if "EmployeeId" in row_df.columns:
            row_df.set_index("EmployeeId", inplace=True)
        else:
            row_df.index = [0]
        
        try:
            x_encoded = encode_single_row(row_df)
            profile = predict_risk_profile(x_encoded)
            
            history_points.append({
                "date": row["valid_from"].split(" ")[0], # YYYY-MM-DD
                "timestamp": row["valid_from"],
                "JobRole": row["JobRole"],
                "MonthlyIncome": float(row["MonthlyIncome"]),
                "OverTime": row["OverTime"],
                "YearsWithCurrManager": int(row["YearsWithCurrManager"]),
                "Prob_12M": f"{profile['Prob_12M'] * 100:.1f}%",
                "General_Risk_Score": round(profile['General_Risk_Score'], 1)
            })
        except Exception as e:
            # Skip invalid history formats silently
            continue
            
    return history_points

@app.post("/api/whatif")
def calculate_whatif(req: WhatIfRequest):
    conn = sqlite3.connect(OLAP_PATH)
    # Fetch active record from Layer 2
    query = "SELECT * FROM v_ml_features WHERE EmployeeId = ?"
    df_active = pd.read_sql(query, conn, params=(req.EmployeeId,))
    conn.close()
    
    if df_active.empty:
        raise HTTPException(status_code=404, detail=f"Employee {req.EmployeeId} not found.")
        
    # Apply overrides
    row_df = df_active.copy()
    if req.MonthlyIncome is not None:
        row_df["MonthlyIncome"] = req.MonthlyIncome
    if req.OverTime is not None:
        row_df["OverTime"] = req.OverTime
    if req.YearsWithCurrManager is not None:
        row_df["YearsWithCurrManager"] = req.YearsWithCurrManager
    if req.JobRole is not None:
        row_df["JobRole"] = req.JobRole
    if req.JobLevel is not None:
        row_df["JobLevel"] = req.JobLevel
        
    # Set EmployeeId as index so it doesn't get coerced to NaN by preprocess_data
    if "EmployeeId" in row_df.columns:
        row_df.set_index("EmployeeId", inplace=True)
        
    # Run transformation and inference
    try:
        x_encoded = encode_single_row(row_df)
        profile = predict_risk_profile(x_encoded)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    return {
        "Prob_1M": f"{profile['Prob_1M'] * 100:.1f}%",
        "Prob_3M": f"{profile['Prob_3M'] * 100:.1f}%",
        "Prob_6M": f"{profile['Prob_6M'] * 100:.1f}%",
        "Prob_12M": f"{profile['Prob_12M'] * 100:.1f}%",
        "General_Risk_Score": round(profile['General_Risk_Score'], 1)
    }

@app.post("/api/simulate-action")
def simulate_action(req: ActionRequest):
    # Import pipeline simulator
    sys.path.append(os.getcwd())
    from pipeline import simulator_actions as sa
    
    try:
        if req.action == "promote":
            if req.params.JobRole is None or req.params.JobLevel is None or req.params.MonthlyIncome is None:
                raise HTTPException(status_code=400, detail="Missing required parameters for promotion")
            sa.promote_employee(
                req.EmployeeId, 
                req.params.JobRole, 
                req.params.JobLevel, 
                float(req.params.MonthlyIncome)
            )
        elif req.action == "overtime":
            if req.params.OverTime is None:
                raise HTTPException(status_code=400, detail="Missing OverTime parameter")
            sa.change_overtime(req.EmployeeId, req.params.OverTime)
        elif req.action == "manager_change":
            if req.params.YearsWithCurrManager is None:
                raise HTTPException(status_code=400, detail="Missing YearsWithCurrManager parameter")
            sa.change_manager_tenure(req.EmployeeId, int(req.params.YearsWithCurrManager))
        elif req.action == "salary_hike":
            if req.params.PercentSalaryHike is None:
                raise HTTPException(status_code=400, detail="Missing PercentSalaryHike parameter")
            sa.adjust_salary(req.EmployeeId, float(req.params.PercentSalaryHike))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")
            
        # Re-run ETL and ML pipelines to commit SCD Type 2 and update score views
        sa.trigger_nightly_etl_ml()
        
        # Reload FastAPI pipeline cache in case baseline or parameters shifted
        load_pipeline()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation transaction failed: {str(e)}")
        
    return {"status": "success", "message": "Simulation action completed and models recalculated."}

@app.get("/api/graph/exposure/{employee_id}")
def get_graph_exposure(employee_id: str):
    """
    Computes Network Exposure score using a Neo4j Cypher query.
    Falls back to a calculated mock score if Neo4j is offline.
    """
    driver = app.state.neo4j_driver
    if not driver:
        # Mock fallback based on generic department risk
        import random
        random.seed(employee_id)
        mock_score = round(random.uniform(0, 3.5), 2)
        mock_peers = [{"id": f"EMP_{random.randint(1000, 1400)}", "role": "Peer", "weight": 0.6}] if mock_score > 1.0 else []
        return {"exposure_score": mock_score, "connected_exits": mock_peers}
        
    query = """
    MATCH (e:Employee {id: $employee_id})-[r:SAME_MANAGER|SAME_ROLE_DEPT|SAME_TENURE_COHORT]-(peer:Employee)
    WHERE peer.isActive = false AND peer.exitDate <> ''
    WITH peer, r, duration.inDays(date(peer.exitDate), date()).days AS days_since_exit
    WHERE days_since_exit <= 60
    RETURN peer.id AS peer_id, peer.jobRole AS peer_role, r.weight AS edge_weight, days_since_exit
    ORDER BY edge_weight DESC LIMIT 5
    """
    
    try:
        with driver.session() as session:
            result = session.run(query, employee_id=employee_id)
            peers = []
            total_exposure = 0.0
            
            for record in result:
                weight = float(record["edge_weight"])
                days = int(record["days_since_exit"])
                # Time decay: fully weighted at day 0, decays to 0 at day 60
                decay = max(0, (60 - days) / 60.0)
                exposure = weight * decay
                total_exposure += exposure
                
                peers.append({
                    "id": record["peer_id"],
                    "role": record["peer_role"],
                    "weight": round(weight, 2),
                    "exposure": round(exposure, 2)
                })
                
            return {
                "exposure_score": round(total_exposure, 2),
                "connected_exits": peers
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j Query Failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Start web server
    uvicorn.run(app, host="127.0.0.1", port=8000)
