"""
simulator_actions.py — Simulated HR administrative actions in oltp_hr.db.

Allows changing employee parameters (OverTime, JobRole, MonthlyIncome)
and triggering the ETL + ML production pipeline to calculate the resulting risks.
"""
import sqlite3
import subprocess
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import OLTP_PATH

def execute_sql(query, params=()):
    conn = sqlite3.connect(OLTP_PATH)
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def promote_employee(employee_id: str, new_role: str, new_level: str, new_income: float):
    print(f"[*] Simulating promotion for {employee_id}: Role={new_role}, Level={new_level}, Income=${new_income}")
    query = """
    UPDATE employees 
    SET JobRole = ?, JobLevel = ?, MonthlyIncome = ? 
    WHERE EmployeeId = ?
    """
    execute_sql(query, (new_role, new_level, new_income, employee_id))
    print("[+] OLTP record updated.")

def change_overtime(employee_id: str, overtime_status: str):
    print(f"[*] Simulating Overtime change for {employee_id}: OverTime={overtime_status}")
    query = "UPDATE employees SET OverTime = ? WHERE EmployeeId = ?"
    execute_sql(query, (overtime_status, employee_id))
    print("[+] OLTP record updated.")

def change_manager_tenure(employee_id: str, years_with_manager: int):
    print(f"[*] Simulating manager change for {employee_id}: YearsWithCurrManager={years_with_manager}")
    query = "UPDATE employees SET YearsWithCurrManager = ? WHERE EmployeeId = ?"
    execute_sql(query, (years_with_manager, employee_id))
    print("[+] OLTP record updated.")

def adjust_salary(employee_id: str, percentage_hike: float):
    conn = sqlite3.connect(OLTP_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MonthlyIncome FROM employees WHERE EmployeeId = ?", (employee_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Employee {employee_id} not found.")
    
    current_income = row[0]
    new_income = round(current_income * (1.0 + percentage_hike / 100.0), 2)
    
    cursor.execute("""
    UPDATE employees 
    SET MonthlyIncome = ?, PercentSalaryHike = ? 
    WHERE EmployeeId = ?
    """, (new_income, percentage_hike, employee_id))
    conn.commit()
    conn.close()
    print(f"[+] Salary adjusted for {employee_id}: ${current_income} -> ${new_income} (+{percentage_hike}%)")

def trigger_nightly_etl_ml():
    """
    Run etl.py then production_ml.py using the correct virtual environment python executable.
    """
    print("[*] Triggering ETL + ML production pipeline run...")
    python_exe = sys.executable
    if not python_exe or "python" not in python_exe.lower():
        # Fallback to local virtualenv python
        python_exe = os.path.join(".venv", "Scripts", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = "python"
            
    print(f"  [ETL] Launching: {python_exe} pipeline/etl.py")
    subprocess.run([python_exe, "pipeline/etl.py"], check=True)
    
    print(f"  [ML] Launching: {python_exe} pipeline/production_ml.py")
    subprocess.run([python_exe, "pipeline/production_ml.py"], check=True)
    
    print("[+] Nightly ETL + ML run finished. Risk scores synchronized.")

if __name__ == "__main__":
    # Test execution
    trigger_nightly_etl_ml()
