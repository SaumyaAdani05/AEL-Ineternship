import sqlite3
import pandas as pd
import os

db_path = os.path.join('data', 'olap_warehouse.db')
conn = sqlite3.connect(db_path)
df = pd.read_sql("SELECT Prob_Leave_12M, General_Risk_Score FROM flight_risk_scores WHERE EmployeeId = 'EMP_0790'", conn)
print(df)
