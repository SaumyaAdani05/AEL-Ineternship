# Implementation Plan: Remove Known Issues and Stabilize Project

This plan lists the practical steps to remove the current technical issues from the project and make the proof-of-concept cleaner.

## Goal

Improve project consistency and reduce future runtime risk without changing the main business logic of the HR attrition dashboard.

## Issues to Remove

| Issue | Current Problem | Target State |
|---|---|---|
| Path hardcoding | `pipeline/simulator_actions.py` uses `OLTP_PATH = "oltp_hr.db"` | Use the correct `data/oltp_hr.db` path from project root |
| Scattered paths | Database and artifact paths are defined in multiple files | Centralize shared paths in one config/helper module |
| Artifact versioning | `.pkl` and `.json` model artifacts have no version metadata | Add model/version metadata file when training runs |
| Dependency drift | `requirements.txt` uses broad minimum versions | Pin tested versions for repeatable setup |
| Production security gap | No auth/audit layer for real MNC deployment | Add authentication, authorization, and audit logging before production use |

## Phase 1: Fix Immediate Path Issue

1. Update `pipeline/simulator_actions.py`.
2. Replace the current relative path:

```python
OLTP_PATH = "oltp_hr.db"
```

with a project-root based path:

```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLTP_PATH = os.path.join(BASE_DIR, "data", "oltp_hr.db")
```

3. Run a quick import check:

```bash
python -m py_compile pipeline/simulator_actions.py
```

4. Test one safe read/query path before running any write simulation.

## Phase 2: Centralize Path Configuration

1. Add or extend a shared config location, preferably `pipeline/config.py`.
2. Define common paths there:

```python
BASE_DIR
DATA_DIR
REPORTS_DIR
OLTP_PATH
OLAP_PATH
MODEL_DIR
```

3. Update these files to use shared paths:

- `pipeline/seed_db.py`
- `pipeline/etl.py`
- `pipeline/production_ml.py`
- `pipeline/simulator_actions.py`
- `app/server.py`
- `run_pipeline.py`

4. Verify all scripts still find files from the project root.

## Phase 3: Add Model Artifact Metadata

1. During `pipeline/production_ml.py`, create a metadata file such as:

```text
pipeline/model_metadata.json
```

2. Store useful values:

- Training timestamp
- Dataset row count
- Feature count
- Model type
- Python version
- Key library versions
- Artifact filenames

3. In `app/server.py`, load and expose this metadata through an optional endpoint:

```text
/api/model/metadata
```

## Phase 4: Pin Dependencies

1. Test the project in the current working environment.
2. Replace broad dependency ranges with exact versions in `requirements.txt`.
3. Keep a note that these versions are the tested proof-of-concept environment.
4. Reinstall in a fresh virtual environment to verify reproducibility.

## Phase 5: Add Validation and Safety Checks

1. Add validation checks before model training:

- Required columns exist
- `Attrition` values are valid
- `YearsAtCompany` is numeric
- No unexpected empty dataset

2. Add stricter request validation in `app/server.py` for simulation actions.
3. Add allowed values for fields like `OverTime`, `JobLevel`, and `JobRole` where possible.

## Phase 6: Production Readiness Items

These are not mandatory for the internship proof-of-concept, but are required before real MNC use:

- Add authentication.
- Add role-based authorization.
- Add audit logs for who viewed or changed risk simulations.
- Move secrets/configuration into environment variables.
- Replace local SQLite with PostgreSQL/MySQL or a managed database.
- Add monitoring for model drift and data quality.
- Add fairness checks for sensitive groups.

## Verification Checklist

After implementation, verify:

- `python pipeline/seed_db.py` runs successfully.
- `python pipeline/etl.py` runs successfully.
- `python pipeline/production_ml.py` runs successfully.
- `python run_pipeline.py` generates reports successfully.
- `python app/server.py` starts the dashboard.
- `/api/employees` returns employee risk rows.
- `/api/dashboard/stats` returns dashboard metrics.
- What-if inference works without changing the database.
- Simulation action uses `data/oltp_hr.db`, not a root-level `oltp_hr.db`.

## Rollback Plan

If any change breaks the app:

1. Restore the previous version of the edited file.
2. Re-run only the last changed script.
3. Check database paths first, because most current risk is path-related.
4. Avoid deleting existing `.db` or model artifact files unless a backup exists.
