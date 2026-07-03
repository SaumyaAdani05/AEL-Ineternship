"""
etl.py — Stage 1 & 2: Slowly Changing Dimension (SCD Type 2) ETL Engine.

Extracts employee records from oltp_hr.db, updates active histories
in the OLAP warehouse (olap_warehouse.db), and partitions them into Layer 1 Theme Tables.
Creates the Layer 2 v_ml_features virtual view.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3
import pandas as pd
from datetime import datetime

from pipeline.config import OLTP_PATH, OLAP_PATH

# Schema mappings for the 5 logical domains
THEME_COLUMNS = {
    "identity": ["EmployeeId", "Age", "Gender", "MaritalStatus", "Education", "EducationField", "BusinessTravel"],
    "environment": ["EmployeeId", "Department", "JobRole", "DistanceFromHome", "OverTime", "JobLevel"],
    "compensation": ["EmployeeId", "MonthlyIncome", "PercentSalaryHike", "StockOptionLevel", "DailyRate", "HourlyRate", "MonthlyRate"],
    "sentiment": ["EmployeeId", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance", "JobInvolvement", "JobSatisfaction", "PerformanceRating"],
    "tenure": ["EmployeeId", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "TotalWorkingYears", "TrainingTimesLastYear", "NumCompaniesWorked", "Attrition"]
}

def init_olap_schema():
    print("[*] Initializing OLAP Warehouse schema...")
    conn = sqlite3.connect(OLAP_PATH)
    cursor = conn.cursor()
    
    # 1. Main history table (SCD Type 2 master log)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employee_history (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        EmployeeId TEXT,
        Age INTEGER, Attrition TEXT, BusinessTravel TEXT, DailyRate REAL,
        Department TEXT, DistanceFromHome INTEGER, Education TEXT,
        EducationField TEXT, EnvironmentSatisfaction TEXT, Gender TEXT,
        HourlyRate REAL, JobInvolvement TEXT, JobLevel TEXT, JobRole TEXT,
        JobSatisfaction TEXT, MaritalStatus TEXT, MonthlyIncome REAL,
        MonthlyRate REAL, NumCompaniesWorked INTEGER, OverTime TEXT,
        PercentSalaryHike REAL, PerformanceRating TEXT, RelationshipSatisfaction TEXT,
        StockOptionLevel INTEGER, TotalWorkingYears INTEGER, TrainingTimesLastYear INTEGER,
        WorkLifeBalance TEXT, YearsAtCompany INTEGER, YearsInCurrentRole INTEGER,
        YearsSinceLastPromotion INTEGER, YearsWithCurrManager INTEGER,
        valid_from TEXT,
        valid_to TEXT,
        is_active INTEGER
    );
    """)
    
    # 2. Theme Tables (Layer 1 Foundation)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS theme_identity (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        EmployeeId TEXT, Age INTEGER, Gender TEXT, MaritalStatus TEXT, 
        Education TEXT, EducationField TEXT, BusinessTravel TEXT,
        valid_from TEXT, valid_to TEXT, is_active INTEGER
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS theme_environment (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        EmployeeId TEXT, Department TEXT, JobRole TEXT, DistanceFromHome INTEGER, 
        OverTime TEXT, JobLevel TEXT,
        valid_from TEXT, valid_to TEXT, is_active INTEGER
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS theme_compensation (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        EmployeeId TEXT, MonthlyIncome REAL, PercentSalaryHike REAL, 
        StockOptionLevel INTEGER, DailyRate REAL, HourlyRate REAL, MonthlyRate REAL,
        valid_from TEXT, valid_to TEXT, is_active INTEGER
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS theme_sentiment (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        EmployeeId TEXT, EnvironmentSatisfaction TEXT, RelationshipSatisfaction TEXT, 
        WorkLifeBalance TEXT, JobInvolvement TEXT, JobSatisfaction TEXT, PerformanceRating TEXT,
        valid_from TEXT, valid_to TEXT, is_active INTEGER
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS theme_tenure (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        EmployeeId TEXT, YearsAtCompany INTEGER, YearsInCurrentRole INTEGER, 
        YearsSinceLastPromotion INTEGER, YearsWithCurrManager INTEGER, 
        TotalWorkingYears INTEGER, TrainingTimesLastYear INTEGER, NumCompaniesWorked INTEGER, Attrition TEXT,
        valid_from TEXT, valid_to TEXT, is_active INTEGER
    );
    """)
    
    # 3. Model score output table (Layer 1 Write-back)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flight_risk_scores (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        EmployeeId TEXT,
        Prob_Leave_1M REAL,
        Prob_Leave_3M REAL,
        Prob_Leave_6M REAL,
        Prob_Leave_12M REAL,
        General_Risk_Score REAL,
        Contrib_Identity REAL,
        Contrib_Environment REAL,
        Contrib_Compensation REAL,
        Contrib_Sentiment REAL,
        Contrib_Tenure REAL,
        DateCalculated TEXT
    );
    """)

    # 4. Create virtual join view (Layer 2 Translation View)
    cursor.execute("DROP VIEW IF EXISTS v_ml_features;")
    cursor.execute("""
    CREATE VIEW v_ml_features AS
    SELECT 
        i.EmployeeId,
        i.Age, i.Gender, i.MaritalStatus, i.Education, i.EducationField, i.BusinessTravel,
        e.Department, e.JobRole, e.DistanceFromHome, e.OverTime, e.JobLevel,
        c.MonthlyIncome, c.PercentSalaryHike, c.StockOptionLevel, c.DailyRate, c.HourlyRate, c.MonthlyRate,
        s.EnvironmentSatisfaction, s.RelationshipSatisfaction, s.WorkLifeBalance, s.JobInvolvement, s.JobSatisfaction, s.PerformanceRating,
        t.YearsAtCompany, t.YearsInCurrentRole, t.YearsSinceLastPromotion, t.YearsWithCurrManager, t.TotalWorkingYears, t.TrainingTimesLastYear, t.NumCompaniesWorked, t.Attrition
    FROM theme_identity i
    JOIN theme_environment e ON i.EmployeeId = e.EmployeeId AND e.is_active = 1
    JOIN theme_compensation c ON i.EmployeeId = c.EmployeeId AND c.is_active = 1
    JOIN theme_sentiment s ON i.EmployeeId = s.EmployeeId AND s.is_active = 1
    JOIN theme_tenure t ON i.EmployeeId = t.EmployeeId AND t.is_active = 1
    WHERE i.is_active = 1;
    """)
    
    conn.commit()
    conn.close()
    print("[+] OLAP Warehouse schema initialized.")

def run_etl():
    """
    Extract from OLTP, compare with OLAP active records, and write SCD Type 2 updates.
    """
    if not os.path.exists(OLTP_PATH):
        raise FileNotFoundError(f"Source OLTP database '{OLTP_PATH}' not found. Run seed_db.py first.")
        
    init_olap_schema()
    
    print("[*] Connecting databases for incremental ETL...")
    conn_oltp = sqlite3.connect(OLTP_PATH)
    conn_olap = sqlite3.connect(OLAP_PATH)
    
    df_oltp = pd.read_sql("SELECT * FROM employees", conn_oltp)
    df_oltp.set_index("EmployeeId", inplace=True)
    
    # Load currently active records in OLAP
    df_olap_active = pd.read_sql("SELECT * FROM employee_history WHERE is_active = 1", conn_olap)
    df_olap_active.set_index("EmployeeId", inplace=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor = conn_olap.cursor()
    
    inserts_count = 0
    updates_count = 0
    deletes_count = 0
    
    # Identify deletions (active in OLAP but missing in OLTP)
    deleted_ids = list(set(df_olap_active.index) - set(df_oltp.index))
    if deleted_ids:
        print(f"[*] Found {len(deleted_ids)} terminated/deleted employees. Deactivating...")
        for emp_id in deleted_ids:
            cursor.execute("UPDATE employee_history SET valid_to = ?, is_active = 0 WHERE EmployeeId = ? AND is_active = 1", (timestamp, emp_id))
            for theme in THEME_COLUMNS.keys():
                cursor.execute(f"UPDATE theme_{theme} SET valid_to = ?, is_active = 0 WHERE EmployeeId = ? AND is_active = 1", (timestamp, emp_id))
            deletes_count += 1
            
    # Iterate OLTP records to find inserts and updates
    for emp_id, row in df_oltp.iterrows():
        # Exclude metadata index
        cols_to_check = [c for c in df_oltp.columns]
        
        if emp_id not in df_olap_active.index:
            # Case A: New Employee
            # 1. Write master history record
            vals = [emp_id] + list(row[cols_to_check]) + [timestamp, None, 1]
            placeholders = ", ".join(["?"] * len(vals))
            cursor.execute(f"INSERT INTO employee_history ({', '.join(['EmployeeId'] + cols_to_check + ['valid_from', 'valid_to', 'is_active'])}) VALUES ({placeholders})", vals)
            
            # 2. Write theme records
            for theme, cols in THEME_COLUMNS.items():
                theme_vals = [emp_id] + [row[c] for c in cols if c != "EmployeeId"] + [timestamp, None, 1]
                theme_placeholders = ", ".join(["?"] * len(theme_vals))
                theme_col_names = ", ".join(cols + ["valid_from", "valid_to", "is_active"])
                cursor.execute(f"INSERT INTO theme_{theme} ({theme_col_names}) VALUES ({theme_placeholders})", theme_vals)
                
            inserts_count += 1
        else:
            # Case B: Check if attributes changed
            active_olap_row = df_olap_active.loc[emp_id]
            changed = False
            for col in cols_to_check:
                # Handle cell value comparison (strings/numeric)
                source_val = row[col]
                dest_val = active_olap_row[col]
                
                # Check for inequality (string type-casted comparisons to avoid floating mismatch noise)
                if str(source_val) != str(dest_val):
                    changed = True
                    break
                    
            if changed:
                # Close out old record version
                cursor.execute("UPDATE employee_history SET valid_to = ?, is_active = 0 WHERE EmployeeId = ? AND is_active = 1", (timestamp, emp_id))
                for theme in THEME_COLUMNS.keys():
                    cursor.execute(f"UPDATE theme_{theme} SET valid_to = ?, is_active = 0 WHERE EmployeeId = ? AND is_active = 1", (timestamp, emp_id))
                
                # Write new active version
                vals = [emp_id] + list(row[cols_to_check]) + [timestamp, None, 1]
                placeholders = ", ".join(["?"] * len(vals))
                cursor.execute(f"INSERT INTO employee_history ({', '.join(['EmployeeId'] + cols_to_check + ['valid_from', 'valid_to', 'is_active'])}) VALUES ({placeholders})", vals)
                
                for theme, cols in THEME_COLUMNS.items():
                    theme_vals = [emp_id] + [row[c] for c in cols if c != "EmployeeId"] + [timestamp, None, 1]
                    theme_placeholders = ", ".join(["?"] * len(theme_vals))
                    theme_col_names = ", ".join(cols + ["valid_from", "valid_to", "is_active"])
                    cursor.execute(f"INSERT INTO theme_{theme} ({theme_col_names}) VALUES ({theme_placeholders})", theme_vals)
                    
                updates_count += 1
                
    conn_olap.commit()
    conn_oltp.close()
    conn_olap.close()
    
    print(f"[+] ETL complete: {inserts_count} inserts, {updates_count} updates, {deletes_count} deletes.")

if __name__ == "__main__":
    run_etl()
