"""
seed_db.py — Initial Database Seeding.

Seeds the simulated Workday/SAP OLTP SQLite database ('oltp_hr.db')
with the cleaned labeled rows from datasets.csv.
"""
import os
import sqlite3
import pandas as pd

# Load raw datasets.csv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "datasets.csv")
OLTP_PATH = os.path.join(BASE_DIR, "data", "oltp_hr.db")

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

    # Write to SQLite
    if os.path.exists(OLTP_PATH):
        os.remove(OLTP_PATH)
        
    print(f"[*] Writing {len(df)} records to '{OLTP_PATH}'...")
    conn = sqlite3.connect(OLTP_PATH)
    df.to_sql("employees", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print("[+] Seeding of OLTP completed successfully.")

if __name__ == "__main__":
    seed_oltp()
