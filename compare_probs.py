
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from pipeline.config import OLAP_PATH, get_conn
from app.server import pipeline_store
from pipeline.production_ml import get_active_cohort
from pipeline.data_pipeline import preprocess_data
from pipeline import config

pipeline_store.reload()

with get_conn(OLAP_PATH) as conn:
    df_scores = pd.read_sql('SELECT EmployeeId, Prob_Leave_12M FROM flight_risk_scores', conn)

df_raw = get_active_cohort()
df_clean = preprocess_data(df_raw)

loo = pipeline_store.get('loo_encoder')
df_encoded = loo.transform(df_clean)

ohe_cols = [c for c in config.ONE_HOT_COLS if c in df_clean.columns]
df_ohe = pd.get_dummies(df_encoded, columns=ohe_cols, drop_first=True)

cols = pipeline_store.get('feature_names')
for c in cols:
    if c not in df_ohe.columns:
        df_ohe[c] = 0
df_ohe = df_ohe[cols].astype(float)

scaler = pipeline_store.get('scaler')
df_ohe[scaler.feature_names_in_] = scaler.transform(df_ohe[scaler.feature_names_in_])

model = pipeline_store.get('model_xgb')
dmat = xgb.DMatrix(df_ohe)
raw_preds = model.predict(dmat, output_margin=True)
risk_multipliers = np.exp(np.clip(raw_preds, -15, 15))

baseline_survival = pipeline_store.get('baseline_survival')
idx = (np.abs(baseline_survival.index - 12)).argmin()
s0_12 = baseline_survival.iloc[idx]

calc_probs = 1.0 - (s0_12 ** risk_multipliers)
df_calc = pd.DataFrame({'EmployeeId': df_ohe.index, 'Calc_Prob': calc_probs, 'RawPred': raw_preds, 'RiskMult': risk_multipliers})

merged = df_calc.merge(df_scores, on='EmployeeId')
merged['Diff'] = np.abs(merged['Calc_Prob'] - merged['Prob_Leave_12M'])
print('Max Diff:', merged['Diff'].max())

emp = merged[merged['EmployeeId'] == 'EMP_0790']
if len(emp) > 0:
    print('EMP_0790 data:')
    for k, v in emp.iloc[0].items():
        print(f'{k}: {v}')
else:
    print('EMP_0790 not found!')

