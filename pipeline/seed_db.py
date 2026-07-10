"""
seed_db.py — Initial Database Seeding.

Seeds the simulated Workday/SAP OLTP SQLite database ('oltp_hr.db')
with the cleaned labeled rows from datasets.csv.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3
import pandas as pd

from datetime import datetime

from pipeline.config import DATA_DIR, OLTP_PATH, generate_performance_score, ORDINAL_MAPPINGS

CSV_PATH = os.path.join(DATA_DIR, "datasets.csv")

def seed_oltp():
    print(f"[*] Reading '{CSV_PATH}' for OLTP seeding...")
    # Load with flexible error handling to match clean labeled boundary
    try:
        df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(CSV_PATH, error_bad_lines=False)
        
    # Standard cleanup of rows matching the main.py preprocessing boundary (approx. 1489 clean rows)
    # The second half of datasets.csv switches formats, so we slice only the first 1489 rows
    df = df.iloc[:1489].copy()
    
    # Generate stable unique EmployeeId
    employee_ids = [f"EMP_{i:04d}" for i in range(1, len(df) + 1)]
    df.insert(0, "EmployeeId", employee_ids)
    
    # Ensure standard types are preserved
    # Convert Yes/No target or categoricals to strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    
    # Fix 'nan' strings in DateOfLeaving back to actual None/NULL
    if 'DateOfLeaving' in df.columns:
        df['DateOfLeaving'] = df['DateOfLeaving'].replace('nan', None)

    # Write to SQLite
    if os.path.exists(OLTP_PATH):
        os.remove(OLTP_PATH)
        
    print(f"[*] Writing {len(df)} records to '{OLTP_PATH}'...")
    conn = sqlite3.connect(OLTP_PATH)
    df.to_sql("employees", conn, if_exists="replace", index=False)
    
    print("[*] Seeding 'performance_ratings' table...")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS performance_ratings (
        EmployeeId TEXT PRIMARY KEY,
        PerformanceScore INTEGER,
        rated_by TEXT,
        rated_at TEXT,
        FOREIGN KEY (EmployeeId) REFERENCES employees(EmployeeId)
    );
    """)
    cursor.execute("DELETE FROM performance_ratings;")
    
    ratings = []
    now = datetime.now().isoformat()
    for _, row in df.iterrows():
        emp_id = row['EmployeeId']
        # Map text values to integers based on config mappings
        js_val = ORDINAL_MAPPINGS["JobSatisfaction"].get(row['JobSatisfaction'], 2)
        ji_val = ORDINAL_MAPPINGS["JobInvolvement"].get(row['JobInvolvement'], 2)
        pr_val = ORDINAL_MAPPINGS["PerformanceRating"].get(row['PerformanceRating'], 2)
        
        score = generate_performance_score(emp_id, js_val, ji_val, pr_val)
        ratings.append((emp_id, score, "system_seed", now))
        
    cursor.executemany("""
        INSERT INTO performance_ratings (EmployeeId, PerformanceScore, rated_by, rated_at)
        VALUES (?, ?, ?, ?)
    """, ratings)
    
    conn.commit()
    conn.close()
    print("[+] Seeding of OLTP completed successfully.")

if __name__ == "__main__":
    seed_oltp()
