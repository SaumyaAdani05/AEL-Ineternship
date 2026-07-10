import pandas as pd
from pipeline.config import OLAP_PATH, get_conn
with get_conn(OLAP_PATH) as conn:
    df = pd.read_sql("SELECT * FROM v_ml_features WHERE EmployeeId='EMP_0790'", conn)
for col in df.columns:
    print(col, df.iloc[0][col])
