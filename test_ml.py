
import sys
sys.path.append('.')
from pipeline.production_ml import get_active_cohort, train_and_save_pipeline, run_predictions_and_writeback
import pandas as pd
import numpy as np
import xgboost as xgb

df_cohort = get_active_cohort()
model, cph, loo_encoder, scaler, df_encoded, X_train, baseline_survival = train_and_save_pipeline(df_cohort)

# Let's see what model predicts for EMP_0790
if 'EMP_0790' in X_train.index:
    dmatrix = xgb.DMatrix(X_train.loc[['EMP_0790']])
    raw_pred = model.predict(dmatrix, output_margin=True)[0]
    clipped = np.clip(raw_pred, -15, 15)
    risk_mult = np.exp(clipped)
    idx = (np.abs(baseline_survival.index - 12)).argmin()
    s0_12 = baseline_survival.iloc[idx]
    prob = 1.0 - (s0_12 ** risk_mult)
    print(f'EMP_0790 raw_pred: {raw_pred}, prob: {prob}')
else:
    print('EMP_0790 not found in X_train!')

