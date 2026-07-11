# HR Attrition Risk Prediction Dashboard

This project is an HR analytics and machine learning application for predicting employee attrition risk. It uses employee data, prepares it through an ETL and ML pipeline, stores risk scores in an analytics warehouse, and shows the results in a FastAPI dashboard.

The project is written in a simple MNC proof-of-concept style. It connects data engineering, survival modelling, backend APIs, graph analytics, and a business dashboard.

## What This Project Does

- Loads HR employee data from `data/datasets.csv`.
- Seeds a simulated HR source database: `data/oltp_hr.db`.
- Builds an analytics warehouse: `data/olap_warehouse.db`.
- Uses SCD Type 2 logic to keep employee history.
- Trains XGBoost survival and Cox PH models.
- Calculates attrition risk for 1 month, 3 months, 6 months, and 12 months.
- Stores model scores in `flight_risk_scores`.
- Evaluates Attrition Contagion Graph (Network Exposure).
- Builds department-level employee similarity networks using manager, department, project, demographic, performance, age, and risk signals.
- Serves a dashboard using FastAPI and HTML.
- Supports employee filtering, performance review, risk review, history, and what-if simulation.

## Project Structure

```text
AEL Internship/
|-- app/
|   |-- server.py
|   |-- dashboard.html
|   |-- org_network_similarity.html
|-- data/
|   |-- datasets.csv
|   |-- oltp_hr.db
|   |-- olap_warehouse.db
|   |-- similarity_network_Human_Resources.json
|   |-- similarity_network_Research_and_Development.json
|   |-- similarity_network_Sales.json
|-- pipeline/
|   |-- config.py
|   |-- seed_db.py
|   |-- etl.py
|   |-- data_pipeline.py
|   |-- graph_exporter.py
|   |-- similarity_network.py
|   |-- synthetic_attributes.py
|   |-- init_neo4j.bat
|   |-- model.py
|   |-- production_ml.py
|   |-- simulator_actions.py
|   |-- visualization.py
|   |-- report.py
|   |-- html_reporter.py
|   |-- scaler.pkl
|   |-- loo_encoder.pkl
|   |-- model_cph.pkl
|   |-- model_xgb.json
|   |-- baseline_survival.pkl
|-- docker-compose.yml
|-- reports/
|   |-- generated/
|   |   |-- report_xgboost.html
|   |   |-- report_cox.html
|   |-- markdown/
|   |   |-- Report_1_Technical_Report.md
|   |   |-- Report_2_Business_Report.md
|   |   |-- images/
|-- legacy/
|-- scratch/
|-- run_pipeline.py
|-- requirements.txt
|-- README.md
```

## Main Components

| Component | Purpose |
|---|---|
| `app/server.py` | FastAPI backend, model inference APIs, simulation jobs, and network endpoints |
| `app/dashboard.html` | Main attrition risk dashboard UI |
| `app/org_network_similarity.html` | Department employee similarity network dashboard |
| `data/datasets.csv` | HR dataset |
| `pipeline/seed_db.py` | Creates the OLTP database and seeds performance ratings |
| `pipeline/etl.py` | Moves source data into OLAP warehouse with SCD Type 2 history |
| `pipeline/synthetic_attributes.py` | Creates deterministic synthetic project assignments |
| `pipeline/similarity_network.py` | Builds department-wise employee similarity graph JSON files |
| `pipeline/production_ml.py` | Trains production models and writes risk scores |
| `pipeline/model.py` | XGBoost survival model and time-horizon probability logic |
| `pipeline/simulator_actions.py` | Applies HR simulation actions and reruns ETL/ML/network refresh steps |
| `run_pipeline.py` | Standalone training/report generation script |
| `reports/markdown/` | Technical and business reports |
| `reports/generated/` | Generated HTML model reports |

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Full Data and ML Pipeline

If you want to rebuild the databases, model scores, and similarity network from the CSV:

```bash
python pipeline/seed_db.py
python pipeline/etl.py
python pipeline/production_ml.py
python -m pipeline.similarity_network
```

This will:

1. Create `data/oltp_hr.db`.
2. Build/update `data/olap_warehouse.db`.
3. Train and save model artifacts in `pipeline/`.
4. Write employee risk scores into `flight_risk_scores`.
5. Generate department similarity graph files in `data/`.

## Run the Dashboard

Start the FastAPI server:

```bash
python app/server.py
```

Then open:

```text
http://127.0.0.1:8000
```

The employee similarity network is available at:

```text
http://127.0.0.1:8000/dashboard/org-network
```

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Opens the dashboard |
| `/api/model/metadata` | GET | Returns model metadata |
| `/api/employees` | GET | Employee list with risk scores, filters, sorting, and pagination |
| `/api/dashboard/stats` | GET | Dashboard summary statistics |
| `/api/employees/{employee_id}/history` | GET | Employee history and historical risk |
| `/api/graph/exposure/{employee_id}` | GET | Neo4j query for contagion exposure, with mock fallback |
| `/api/whatif` | POST | Temporary what-if risk calculation |
| `/api/simulate-action` | POST | Applies simulated HR action and reruns ETL/ML |
| `/api/simulate-action/{job_id}` | GET | Polls a background simulation job |
| `/api/retrain` | POST | Starts a retraining job |
| `/api/employees/{employee_id}/rating` | PATCH | Updates a manager performance rating |
| `/api/org-network/tree` | GET | Returns organization hierarchy data |
| `/api/similarity-network/{department}` | GET | Returns pre-generated similarity network data for a department |
| `/api/org-network/exposure-history/{employee_id}` | GET | Returns employee exposure history for the org network view |
| `/dashboard/org-network` | GET | Opens the employee similarity network dashboard |

## Attrition Contagion Graph (Network Exposure)

An investigation was conducted to determine if attrition clusters temporally within departments, meaning whether one employee leaving triggers peers to leave.

1. **Validation (Phase 0):** A robust Monte Carlo permutation test over the historical database proved the actual lift was only `0.95x`. This statistically confirmed that exits do **not** cluster.
2. **Infrastructure Built:** Although not integrated into the ML pipeline due to the negative signal, the graph infrastructure was fully built. Synthetic hierarchical data is generated via `pipeline/graph_exporter.py` and can be imported into a local Neo4j Community instance or via `docker-compose`.
3. **Dashboard Fallback:** The FastAPI server connects to Neo4j to query time-decayed exposure scores. If the instance is offline, it gracefully falls back to mock visual scores in the frontend Employee Detail drawer. The `production_ml.py` model explicitly excludes these scores to prevent noise.

## Employee Similarity Network

The project also includes a department-scoped employee similarity network. This view uses pre-generated JSON files from `pipeline/similarity_network.py` and connects employees who are most similar within the same department.

Similarity is based on:

- Hierarchical context: manager, department, and synthetic project.
- Categorical attributes: gender and performance tier.
- Continuous attributes: age and current attrition risk score.

Synthetic project assignments are generated by `pipeline/synthetic_attributes.py` and are deterministic across ETL runs. The similarity generator writes one file per department:

- `data/similarity_network_Human_Resources.json`
- `data/similarity_network_Research_and_Development.json`
- `data/similarity_network_Sales.json`

When a simulation action triggers the nightly ETL/ML workflow, similarity network generation is also run so the org network dashboard stays aligned with the latest warehouse state.

## Model Flow

The model follows a survival analysis approach. This is used because attrition is time-based.

Basic flow:

1. Clean HR data.
2. Encode categorical columns.
3. Scale numeric columns.
4. Train XGBoost with `survival:cox`.
5. Estimate baseline survival with Nelson-Aalen.
6. Shift survival curve for each employee based on risk.
7. Calculate leave probability at 1M, 3M, 6M, and 12M.

*(Note: The What-If simulation engine incorporates a business rule heuristic to penalize salary cuts, addressing the XGBoost model's limitation since historical data lacked negative salary increments).*

Simple formula:

```text
Individual Survival = Baseline Survival ^ Risk Multiplier
Attrition Probability = 1 - Individual Survival
```

## Current Data Snapshot

Current warehouse snapshot:

| Metric | Value |
|---|---:|
| Active employees | 1,489 |
| Scored employees | 1,489 |
| Average 12-month risk | 49.19% |
| High-risk employees above 30% threshold | 809 |

Department view:

| Department | Employees | Average 12-Month Risk | High-Risk Employees |
|---|---:|---:|---:|
| Research & Development | 975 | 49.65% | 543 |
| Sales | 451 | 48.00% | 229 |
| Human Resources | 63 | 50.62% | 37 |

## Reports

Two markdown reports are available:

- `reports/markdown/Report_1_Technical_Report.md`
- `reports/markdown/Report_2_Business_Report.md`

Generated HTML reports are available in:

- `reports/generated/report_xgboost.html`
- `reports/generated/report_cox.html`

Report images and diagrams are inside:

- `reports/markdown/images/`

## Recent Updates

- **Employee Similarity Network:** Added a department-level org network dashboard backed by precomputed k-nearest-neighbor similarity graphs.
- **Synthetic Project Attribute:** Added deterministic project assignment during ETL so employees can be grouped by a realistic project-like signal even though the original dataset does not contain a project column.
- **Performance Rating Workflow:** Added manager performance rating support and filtering, with ratings used as one signal in employee similarity calculations.
- **Neo4j Connectivity:** Configured the application to support standalone local Neo4j instances as an alternative to Docker.
- **ML What-If Heuristics:** Injected a dynamic penalty multiplier into the What-If simulation engine to model increased attrition risk associated with salary cuts, addressing the historical training data's lack of negative increments.
- **Org Graph UI:** Improved the Org Network Dashboard's mouse cursors to keep them readable on the light background theme.

## Business Use

The dashboard can help HR teams and business heads:

- Identify high-risk employees early.
- See risk by department.
- Understand whether risk is related to compensation, environment, sentiment, tenure, performance, or identity factors.
- Explore employee similarity clusters by department.
- Try what-if actions before applying them.
- Support retention planning with data.

## Disclaimer

This project should be used as a decision-support tool, not as the final decision maker. HR teams should use model output carefully and combine it with human judgement, policy, and ethical review.
