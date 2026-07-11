import sys
sys.path.append('.')
from pipeline.config import OLAP_PATH, get_conn
import pandas as pd
import numpy as np
import xgboost as xgb
from app.server import pipeline_store, encode_single_row
from pipeline.data_pipeline import preprocess_data

pipeline_store.reload()
model = pipeline_store.get('model_xgb')
with get_conn(OLAP_PATH) as conn:
    df = pd.read_sql("SELECT * FROM v_ml_features WHERE EmployeeId='EMP_0790'", conn)
df_clean = preprocess_data(df)
x_encoded = encode_single_row(df_clean)

dmat = xgb.DMatrix(x_encoded)
raw_pred = model.predict(dmat, output_margin=True)[0]
print('Server raw_pred:', raw_pred)

print(x_encoded.T)
