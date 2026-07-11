import sys
sys.path.append('.')
from pipeline.production_ml import get_active_cohort, train_and_save_pipeline
from app.server import pipeline_store, encode_single_row
import pandas as pd
from pipeline.config import OLAP_PATH, get_conn

# 1. Get X_train from production
df_cohort = get_active_cohort()
_, _, _, _, _, X_train, _ = train_and_save_pipeline(df_cohort)
x_train_emp = X_train.loc[['EMP_0790']]

# 2. Get x_encoded from server
with get_conn(OLAP_PATH) as conn:
    df_raw = pd.read_sql("SELECT * FROM v_ml_features WHERE EmployeeId='EMP_0790'", conn)
pipeline_store.reload()
# Must set index for server just like UI endpoint does
if 'EmployeeId' in df_raw.columns:
    df_raw.set_index('EmployeeId', inplace=True)
else:
    df_raw.index = [0]
x_encoded = encode_single_row(df_raw)

print('Feature Differences:')
diff_count = 0
for col in x_train_emp.columns:
    val_train = float(x_train_emp.loc['EMP_0790', col])
    val_server = float(x_encoded.loc[0, col])
    if abs(val_train - val_server) > 1e-5:
        print(f'{col}: train={val_train}, server={val_server}')
        diff_count += 1
if diff_count == 0:
    print('No differences found!')
