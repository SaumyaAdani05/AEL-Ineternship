import sqlite3
import json
import random
import os

DB_PATH = 'data/olap_warehouse.db'
OUTPUT_PATH = 'data/org_hierarchy.json'

def generate_hierarchy():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get the latest snapshot for every distinct employee
    # We use row_number() over (partition by EmployeeId order by valid_from desc)
    query = """
    SELECT EmployeeId, Department, JobRole, JobLevel, is_active
    FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY EmployeeId ORDER BY valid_from DESC) as rn
        FROM employee_history
    )
    WHERE rn = 1
    """
    
    employees = cursor.execute(query).fetchall()
    conn.close()

    job_level_ranks = {
        'Executive Level': 5,
        'Senior Level': 4,
        'Mid Level': 3,
        'Junior Level': 2,
        'Entry Level': 1
    }

    # Group employees by department
    dept_map = {}
    for emp in employees:
        dept = emp['Department']
        if not dept:
            dept = 'Unknown'
        if dept not in dept_map:
            dept_map[dept] = []
        dept_map[dept].append({
            'id': emp['EmployeeId'],
            'department': dept,
            'role': emp['JobRole'],
            'level': job_level_ranks.get(emp['JobLevel'], 1),
            'is_active': emp['is_active']
        })

    nodes = []
    
    # 1. Add CEO
    nodes.append({
        'id': 'CEO',
        'manager_id': '',
        'department': 'Executive',
        'role': 'Chief Executive Officer',
        'is_active': 1
    })

    # 2. Add Dept Heads and build hierarchy per dept
    for dept, emps in dept_map.items():
        head_id = f"HEAD_{dept.replace(' ', '_')}"
        nodes.append({
            'id': head_id,
            'manager_id': 'CEO',
            'department': dept,
            'role': f'VP of {dept}',
            'is_active': 1
        })
        
        # Sort employees by level descending so we assign managers from top down
        emps.sort(key=lambda x: x['level'], reverse=True)
        
        # We need a pool of potential managers at each level
        level_pools = {1:[], 2:[], 3:[], 4:[], 5:[]}
        for emp in emps:
            level_pools[emp['level']].append(emp['id'])
            
        for emp in emps:
            my_level = emp['level']
            # Find a manager in a higher level
            manager_id = None
            for higher_level in range(my_level + 1, 6):
                if level_pools[higher_level]:
                    manager_id = random.choice(level_pools[higher_level])
                    break
            
            if not manager_id:
                manager_id = head_id
                
            nodes.append({
                'id': emp['id'],
                'manager_id': manager_id,
                'department': emp['department'],
                'role': emp['role'],
                'is_active': emp['is_active']
            })

    # Ensure output dir exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump({'nodes': nodes}, f, indent=2)

    print(f"Generated {len(nodes)} nodes.")
    print(f"CEO + {len(dept_map)} departments + {sum(len(e) for e in dept_map.values())} employees.")

if __name__ == "__main__":
    generate_hierarchy()
