"""
production_ml.py — Production ML Engine for Attrition Prediction.

Queries v_ml_features, trains the XGBoost Survival and Cox PH models,
saves model checkpoints, calculates multi-horizon probabilities and theme risk contributions,
and writes scores back to the OLAP database.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from datetime import datetime

# Reconfigure stdout to support unicode symbols
sys.stdout.reconfigure(encoding='utf-8')


from pipeline import config
from pipeline.data_pipeline import preprocess_data
from pipeline.model import prepare_survival_target, estimate_baseline_survival
from pipeline import cli_formatter as cf

from pipeline.config import OLAP_PATH, MODEL_DIR

def get_active_cohort() -> pd.DataFrame:
    print("[*] Querying active employee cohort from Layer 2 view...")
    conn = sqlite3.connect(OLAP_PATH)
    df = pd.read_sql("SELECT * FROM v_ml_features", conn)
    conn.close()
    
    # Set EmployeeId as index
    df.set_index("EmployeeId", inplace=True)
    
    if "DateOfLeaving" in df.columns:
        df.drop(columns=["DateOfLeaving"], inplace=True)
        
    print(f"  [OK] Retracted {len(df)} active employee records.")
    return df

def train_and_save_pipeline(df: pd.DataFrame):
    print("[*] Preprocessing and encoding active dataset...")
    # Preprocess (maps ordinals, drops constants, drops NaNs)
    df_clean = preprocess_data(df)
    
    # Prepare target
    _, y = prepare_survival_target(df_clean)

    
    # Perform nominal encoding
    # In production, we fit encoders on the entire clean dataset
    ohe_cols = [c for c in config.ONE_HOT_COLS if c in df_clean.columns]
    df_encoded = pd.get_dummies(df_clean, columns=ohe_cols, drop_first=True)
    
    # Convert bool columns to numeric (from dummy variables)
    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(int)
        
    loo_cols = [c for c in config.LOO_COLS if c in df_encoded.columns]
    loo_encoder = ce.LeaveOneOutEncoder(cols=loo_cols)
    df_encoded = loo_encoder.fit_transform(df_encoded, df_clean[config.EVENT_COL])
    
    # Fit MinMaxScaler on numeric columns
    scaler = MinMaxScaler()
    exclude_cols = {config.DURATION_COL, config.EVENT_COL}
    num_cols = [
        col
        for col in df_encoded.columns
        if col not in exclude_cols and df_encoded[col].dtype in [np.float64, np.int64, np.int32, np.float32, int, float]
    ]
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

    
    # 1. Fit Cox PH Model (for Explainability & Theme Contributions)
    print("[*] Fitting Cox Proportional Hazards model...")
    cph_df = df_encoded.drop(columns=[config.DURATION_COL]) # Lifelines takes duration separately or in df
    cph_df[config.DURATION_COL] = df_clean[config.DURATION_COL]
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cph_df, duration_col=config.DURATION_COL, event_col=config.EVENT_COL)
    print(f"  [OK] Cox PH trained. C-Index: {cph.concordance_index_:.3f}")
    
    # 2. Fit XGBoost Survival Model (for Predictive Accuracy)
    print("[*] Fitting XGBoost Survival model...")
    X_train = df_encoded.drop(columns=[config.DURATION_COL, config.EVENT_COL])
    
    # Convert bool columns to numeric (from dummy variables)
    for col in X_train.select_dtypes(include=['bool']).columns:
        X_train[col] = X_train[col].astype(int)
        
    dtrain = xgb.DMatrix(X_train, label=y)
    
    params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "learning_rate": 0.05,
        "max_depth": 4,
        "seed": config.RANDOM_STATE
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=120
    )
    print("  [OK] XGBoost Survival trained.")
    
    # 3. Estimate Baseline Survival Curve
    print("[*] Estimating baseline survival curve...")
    _, baseline_survival = estimate_baseline_survival(df_clean)

    
    # Save Pipeline artifacts
    print(f"[*] Saving pipeline checkpoints to '{MODEL_DIR}'...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "loo_encoder.pkl"), "wb") as f:
        pickle.dump(loo_encoder, f)
    with open(os.path.join(MODEL_DIR, "model_cph.pkl"), "wb") as f:
        pickle.dump(cph, f)
        
    model.save_model(os.path.join(MODEL_DIR, "model_xgb.json"))
    
    # Save baseline survival helper
    with open(os.path.join(MODEL_DIR, "baseline_survival.pkl"), "wb") as f:
        pickle.dump(baseline_survival, f)
        
    # Save model metadata
    import json
    metadata = {
        "training_timestamp": datetime.now().isoformat(),
        "dataset_rows": len(df_clean),
        "features_count": len(X_train.columns),
        "model_type": "XGBoost Survival & Cox PH",
        "python_version": sys.version,
        "xgboost_version": xgb.__version__,
        "pandas_version": pd.__version__,
        "artifacts": [
            "scaler.pkl", 
            "loo_encoder.pkl", 
            "model_cph.pkl", 
            "model_xgb.json", 
            "baseline_survival.pkl"
        ]
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("[+] Model artifacts and metadata saved successfully.")
    return model, cph, loo_encoder, scaler, df_encoded, X_train, baseline_survival

def compute_theme_contributions(cph, df_encoded: pd.DataFrame) -> pd.DataFrame:
    """
    Computes theme-level risk contributions using Cox PH coefficients.
    """
    params = cph.params_
    
    theme_mapping = {}
    for col in df_encoded.columns:
        col_lower = col.lower()
        if col_lower in ['age', 'education', 'businesstravel'] or col_lower.startswith('gender') or col_lower.startswith('maritalstatus') or col_lower.startswith('educationfield'):
            theme_mapping[col] = 'identity'
        elif col_lower in ['distancefromhome', 'joblevel'] or col_lower.startswith('department') or col_lower.startswith('overtime') or col_lower.startswith('jobrole'):
            theme_mapping[col] = 'environment'
        elif col_lower in ['monthlyincome', 'percentsalaryhike', 'stockoptionlevel', 'dailyrate', 'hourlyrate', 'monthlyrate']:
            theme_mapping[col] = 'compensation'
        elif col_lower in ['environmentsatisfaction', 'relationshipsatisfaction', 'worklifebalance', 'jobinvolvement', 'jobsatisfaction', 'performancerating']:
            theme_mapping[col] = 'sentiment'
        elif col_lower in ['yearsatcompany', 'yearsincurrentrole', 'yearssincelastpromotion', 'yearswithcurrmanager', 'totalworkingyears', 'trainingtimeslastyear', 'numcompaniesworked']:
            theme_mapping[col] = 'tenure'
        else:
            theme_mapping[col] = 'tenure'
            
    contributions = []
    
    for idx, row in df_encoded.iterrows():
        theme_sums = {'identity': 0.0, 'environment': 0.0, 'compensation': 0.0, 'sentiment': 0.0, 'tenure': 0.0}
        
        # Calculate positive beta * feature contributions
        for feat, val in row.items():
            if feat in params.index:
                log_haz = params[feat] * val
                theme = theme_mapping.get(feat, 'tenure')
                if log_haz > 0:
                    theme_sums[theme] += log_haz
                    
        total_positive = sum(theme_sums.values())
        if total_positive > 0:
            theme_pcts = {t: (v / total_positive) * 100 for t, v in theme_sums.items()}
        else:
            theme_pcts = {t: 20.0 for t in theme_sums.keys()}
            
        contributions.append({
            "EmployeeId": idx,
            "Contrib_Identity": theme_pcts['identity'],
            "Contrib_Environment": theme_pcts['environment'],
            "Contrib_Compensation": theme_pcts['compensation'],
            "Contrib_Sentiment": theme_pcts['sentiment'],
            "Contrib_Tenure": theme_pcts['tenure']
        })
        
    return pd.DataFrame(contributions).set_index("EmployeeId")

def run_predictions_and_writeback(model, cph, loo_encoder, scaler, df_encoded, X_train, baseline_survival):
    print("[*] Calculating time-horizon risk probabilities for active employees...")
    
    # Predict margins/multipliers
    dmatrix = xgb.DMatrix(X_train)
    raw_predictions = model.predict(dmatrix, output_margin=True)
    # Clip predictions to [-15, 15] to prevent exponential overflow
    clipped_predictions = np.clip(raw_predictions, -15, 15)
    risk_multipliers = np.exp(clipped_predictions)

    
    # Calculate probabilities at 1, 3, 6, and 12 months
    horizons = [1, 3, 6, 12]
    probs = {}
    
    for h in horizons:
        # Get baseline survival rate at month h
        # Find closest index in baseline curve index
        idx = (np.abs(baseline_survival.index - h)).argmin()
        s0_t = baseline_survival.iloc[idx]
        
        # S_i(t) = S_0(t) ^ risk_multiplier
        # Prob_Leave = 1 - S_i(t)
        probs[h] = 1.0 - (s0_t ** risk_multipliers)
        
    # Build results dataframe
    results = pd.DataFrame({
        "EmployeeId": X_train.index,
        "Prob_Leave_1M": probs[1],
        "Prob_Leave_3M": probs[3],
        "Prob_Leave_6M": probs[6],
        "Prob_Leave_12M": probs[12],
        "General_Risk_Score": probs[12] * 100 # Scaled 12M probability
    })
    
    # Compute theme risk contributions
    contributions_df = compute_theme_contributions(cph, X_train)
    
    # Join predictions and contributions
    final_scores = results.join(contributions_df, on="EmployeeId")
    final_scores["DateCalculated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write back to database
    print(f"[*] Writing {len(final_scores)} risk score rows back to flight_risk_scores table...")
    conn = sqlite3.connect(OLAP_PATH)
    
    # We clear old calculations first to keep active list clean
    cursor = conn.cursor()
    cursor.execute("DELETE FROM flight_risk_scores;")
    conn.commit()
    
    final_scores.to_sql("flight_risk_scores", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print("[+] ML Inference scores written back successfully.")

def execute_ml_pipeline():
    df_cohort = get_active_cohort()
    model, cph, loo_encoder, scaler, df_encoded, X_train, baseline_survival = train_and_save_pipeline(df_cohort)
    run_predictions_and_writeback(model, cph, loo_encoder, scaler, df_encoded, X_train, baseline_survival)

if __name__ == "__main__":
    execute_ml_pipeline()
