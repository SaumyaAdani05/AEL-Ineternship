import sqlite3
import pandas as pd
import hashlib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import OLAP_PATH

def hash_employee(emp_id: str) -> str:
    return hashlib.md5(emp_id.encode()).hexdigest()

def build_graph_csvs():
    print("[*] Connecting to OLAP Warehouse...")
    conn = sqlite3.connect(OLAP_PATH)
    
    # We want the most recent state for each employee (active or not)
    query = """
    SELECT 
        EmployeeId, Department, JobRole, JobLevel, YearsAtCompany, 
        is_active, DateOfLeaving
    FROM employee_history
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Sort by valid_from (implicitly by row_id descending if we just drop duplicates)
    # Actually, let's just use the latest version of each employee
    df = df.drop_duplicates(subset=['EmployeeId'], keep='last').copy()
    
    print(f"[*] Found {len(df)} total employees for the graph.")
    
    # Precompute tenure cohort
    def get_tenure_cohort(years):
        if years < 2: return "0-2"
        if years < 5: return "2-5"
        if years < 10: return "5-10"
        return "10+"
    
    df['tenureCohort'] = df['YearsAtCompany'].apply(get_tenure_cohort)
    df['isActive'] = df['is_active'] == 1
    
    # 1. Generate Nodes
    print("[*] Generating nodes.csv...")
    nodes = df[['EmployeeId', 'Department', 'JobRole', 'JobLevel', 'tenureCohort', 'isActive', 'DateOfLeaving']].copy()
    nodes.columns = ['id:ID', 'department', 'jobRole', 'jobLevel', 'tenureCohort', 'isActive:boolean', 'exitDate']
    nodes['exitDate'] = nodes['exitDate'].fillna('')
    nodes.to_csv("data/nodes.csv", index=False)
    
    # 2. Fabricate Synthetic Managers and generate SAME_MANAGER edges
    print("[*] Fabricating Synthetic Managers and edges...")
    
    # Create hash for sorting
    df['hash'] = df['EmployeeId'].apply(hash_employee)
    
    manager_edges = []
    
    # Bucket into groups of 8
    grouped = df.groupby(['Department', 'JobRole', 'JobLevel'])
    for name, group in grouped:
        # Sort by hash
        group_sorted = group.sort_values('hash')
        emp_ids = group_sorted['EmployeeId'].tolist()
        
        # Chunk into size 8
        for i in range(0, len(emp_ids), 8):
            chunk = emp_ids[i:i+8]
            # Create a clique for SAME_MANAGER among the chunk
            for emp1 in chunk:
                for emp2 in chunk:
                    if emp1 != emp2:
                        manager_edges.append((emp1, emp2, 1.0, 'inferred', 'SAME_MANAGER'))
                        
    df_manager = pd.DataFrame(manager_edges, columns=[':START_ID', ':END_ID', 'weight:float', 'source', ':TYPE'])
    df_manager.to_csv("data/edges_manager.csv", index=False)
    
    # 3. Generate SAME_ROLE_DEPT edges
    print("[*] Generating SAME_ROLE_DEPT edges...")
    # These are too massive if we fully connect a department. 
    # Let's only connect people with the EXACT same JobRole and Department.
    role_edges = []
    for name, group in df.groupby(['Department', 'JobRole']):
        # If the group is large, a full clique is too many edges (e.g., 500 sales execs = 250,000 edges)
        # We will create a small-world or just sequential connection to keep graph size reasonable,
        # OR fully connect if group size < 50. Let's do fully connected but sample if it's huge.
        emp_ids = group['EmployeeId'].tolist()
        # Cap fully connected clique to limit edge explosion
        if len(emp_ids) > 100:
            import random
            random.seed(42)
            # Create a random geometric/Erdos-Renyi graph approx 5 edges per node
            for e1 in emp_ids:
                for _ in range(5):
                    e2 = random.choice(emp_ids)
                    if e1 != e2:
                        role_edges.append((e1, e2, 0.6, 'inferred', 'SAME_ROLE_DEPT'))
        else:
            for e1 in emp_ids:
                for e2 in emp_ids:
                    if e1 != e2:
                        role_edges.append((e1, e2, 0.6, 'inferred', 'SAME_ROLE_DEPT'))
                        
    df_role = pd.DataFrame(role_edges, columns=[':START_ID', ':END_ID', 'weight:float', 'source', ':TYPE']).drop_duplicates()
    df_role.to_csv("data/edges_role.csv", index=False)
    
    # 4. Generate SAME_TENURE_COHORT edges (also sampled if large)
    print("[*] Generating SAME_TENURE_COHORT edges...")
    tenure_edges = []
    for name, group in df.groupby('tenureCohort'):
        emp_ids = group['EmployeeId'].tolist()
        import random
        random.seed(42)
        for e1 in emp_ids:
            for _ in range(3):
                e2 = random.choice(emp_ids)
                if e1 != e2:
                    tenure_edges.append((e1, e2, 0.4, 'inferred', 'SAME_TENURE_COHORT'))
                    
    df_tenure = pd.DataFrame(tenure_edges, columns=[':START_ID', ':END_ID', 'weight:float', 'source', ':TYPE']).drop_duplicates()
    df_tenure.to_csv("data/edges_tenure.csv", index=False)
    
    print(f"[+] Graph generation complete.")
    print(f"    Nodes: {len(nodes)}")
    print(f"    Manager Edges: {len(df_manager)}")
    print(f"    Role Edges: {len(df_role)}")
    print(f"    Tenure Edges: {len(df_tenure)}")

if __name__ == '__main__':
    build_graph_csvs()
