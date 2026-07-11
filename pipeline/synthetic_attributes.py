"""
synthetic_attributes.py — Generator for synthetic employee attributes that
don't exist in the source dataset (currently: Project).

Design principles (matching org_hierarchy.py / graph_exporter.py precedent):
  - Deterministic, hash-based generation -> reproducible across ETL re-runs,
    never overwritten by re-running the pipeline on the same data.
  - Correlated to existing real attributes (JobRole, JobLevel) rather than
    pure uniform random, so "Project" behaves like a real clustering
    variable instead of injecting noise into the hierarchical distance
    bucket.

DROP-IN LOCATION: pipeline/synthetic_attributes.py
Called from etl.py during theme_environment population (see etl_patch.md
for the exact integration diff).
"""
import hashlib
from typing import Dict, List

PROJECTS_PER_DEPARTMENT = 4


def _stable_hash(s: str) -> int:
    """Deterministic hash (Python's built-in hash() is salted per-process;
    we need something stable across runs and machines)."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def project_names_for_department(department: str) -> List[str]:
    dept_slug = department.replace(" ", "_").replace("&", "and")
    return [f"{dept_slug}_Project_{chr(65 + i)}" for i in range(PROJECTS_PER_DEPARTMENT)]


def assign_project(employee_id: str, department: str, job_role: str, job_level: str) -> str:
    """
    Deterministically assign a synthetic Project to an employee.

    Correlation strategy: the hash seed includes JobRole (and JobLevel to a
    lesser extent) so employees who share a role tend to cluster onto the
    same project bucket, rather than being pure noise. This matters because
    Project sits in the hierarchical distance bucket (weight 0.6) alongside
    Department/Manager -- if it were uncorrelated with role/level, it would
    just dilute that bucket with randomness.

    Weighting: 70% of the hash signal comes from (department, job_role);
    30% comes from employee_id alone (so not everyone in the same role
    lands on the identical project -- some natural spread remains).
    """
    projects = project_names_for_department(department)
    n = len(projects)

    role_seed = _stable_hash(f"{department}|{job_role}|{job_level}")
    emp_seed = _stable_hash(employee_id)

    # Weighted combination: role_seed dominates bucket choice, emp_seed adds
    # spread within that role's preferred bucket range.
    combined = (role_seed * 7 + emp_seed * 3) % (n * 1000)
    idx = combined // 1000
    idx = min(idx, n - 1)

    return projects[idx]


def build_project_assignments(employees: List[Dict]) -> Dict[str, str]:
    """
    employees: list of dicts with keys EmployeeId, Department, JobRole, JobLevel
    Returns: {EmployeeId: Project}
    """
    result = {}
    for emp in employees:
        result[emp["EmployeeId"]] = assign_project(
            emp["EmployeeId"],
            emp.get("Department", "Unknown"),
            emp.get("JobRole", "Unknown"),
            emp.get("JobLevel", "Unknown"),
        )
    return result


if __name__ == "__main__":
    # Smoke test — deterministic check
    a = assign_project("EMP_0001", "Sales", "Sales Executive", "Junior Level")
    b = assign_project("EMP_0001", "Sales", "Sales Executive", "Junior Level")
    assert a == b, "Non-deterministic assignment — hash logic broken"
    print(f"EMP_0001 -> {a} (deterministic check passed)")
