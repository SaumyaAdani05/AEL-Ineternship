
import os, time
from pipeline.config import OLAP_PATH, MODEL_DIR, get_conn
import pandas as pd
print('DB modified:', time.ctime(os.path.getmtime(OLAP_PATH)))
print('XGB modified:', time.ctime(os.path.getmtime(os.path.join(MODEL_DIR, 'model_xgb.json'))))

with get_conn(OLAP_PATH) as conn:
    df_scores = pd.read_sql('SELECT * FROM flight_risk_scores WHERE EmployeeId = \'EMP_0790\'', conn)
print(df_scores[['Prob_Leave_12M']])

