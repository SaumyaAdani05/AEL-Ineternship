"""
run_pipeline.py — Entry Point for the XGBoost Survival Pipeline.

Orchestrates the full 3-stage architecture:
  Stage 1: Build the Unified Data Distribution (encoding + scaling)
  Stage 2: Train XGBoost with survival:cox objective
  Stage 3: Time-Shift → per-employee time-horizon probabilities

Then generates visualizations and executive report.
"""
import sys
import warnings
from pathlib import Path
from pipeline.config import DATA_DIR, REPORTS_DIR

import numpy as np

# Ensure proper encoding for terminal output
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pipeline.data_pipeline import build_unified_distribution
from pipeline.model import (
    estimate_baseline_survival,
    evaluate_model,
    predict_time_horizon_probabilities,
    prepare_survival_target,
    train_xgboost_survival,
)
from pipeline.report import print_executive_report, print_risk_profiles
from pipeline.visualization import plot_pipeline_results
from pipeline import cli_formatter as cf
from pipeline.html_reporter import generate_xgboost_html_report




def main() -> None:
    # Resolve dataset path from config
    dataset_path = Path(DATA_DIR) / "datasets.csv"

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    # ================================================================
    #  STAGE 1: BUILD THE UNIFIED DATA DISTRIBUTION
    # ================================================================
    train_data, test_data, scaler, loo_encoder = build_unified_distribution(
        str(dataset_path)
    )

    # ================================================================
    #  STAGE 2: TRAIN XGBOOST SURVIVAL MODEL
    # ================================================================
    X_train, y_train = prepare_survival_target(train_data)
    X_test, y_test = prepare_survival_target(test_data)

    cf.init_terminal()
    title2 = cf.style("STAGE 2: TRAINING XGBOOST SURVIVAL ENGINE", cf.CYAN, cf.BOLD)
    print("\n" + cf.make_panel(title2, [
        f"Input Model Features: {X_train.shape[1]} columns",
        f"Training Sample Count: {len(X_train)} employees",
        "Orchestrating training with early stopping on 15% validation split..."
    ], style_color=cf.CYAN, width=82))

    # Use a portion of training data for early stopping validation
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    model = train_xgboost_survival(X_tr, y_tr, X_val, y_val)

    # Evaluate
    train_cindex = evaluate_model(model, X_train, train_data)
    test_cindex = evaluate_model(model, X_test, test_data)

    stage2_done_lines = [
        "XGBoost Survival Engine Model trained successfully!",
        "---",
        f"  {cf.OK_ICON}  Training C-Index: {cf.style(f'{train_cindex * 100:.1f}%', cf.GREEN, cf.BOLD)}",
        f"  {cf.OK_ICON}  Test C-Index:     {cf.style(f'{test_cindex * 100:.1f}%', cf.GREEN, cf.BOLD)}"
    ]
    print("\n" + cf.make_panel(cf.style("STAGE 2 COMPLETED", cf.GREEN, cf.BOLD), stage2_done_lines, style_color=cf.GREEN, width=82) + "\n")

    # ================================================================
    #  STAGE 3: TIME-SHIFT MECHANISM
    # ================================================================
    title3 = cf.style("STAGE 3: COMPUTING TIME-HORIZON PROBABILITIES", cf.CYAN, cf.BOLD)
    print("\n" + cf.make_panel(title3, [
        "Computing time-horizon cumulative risk probabilities...",
        f"Estimating baseline hazard from {len(train_data)} training employees."
    ], style_color=cf.CYAN, width=82))

    # Estimate baseline survival from training data
    naf, baseline_survival = estimate_baseline_survival(train_data)

    # Compute per-employee probabilities for test set
    risk_table = predict_time_horizon_probabilities(
        model, baseline_survival, X_test
    )

    # Sanity checks
    assert risk_table.shape[1] == 4, "Expected 4 time horizon columns"
    assert (risk_table >= 0).all().all(), "Probabilities should be >= 0"
    assert (risk_table <= 1).all().all(), "Probabilities should be <= 1"

    # Check monotonicity (probabilities should increase over time)
    monotonic_check = (risk_table.diff(axis=1).iloc[:, 1:] >= -0.01).all().all()
    
    stage3_lines = [
        f"Per-employee risk profiles generated for {len(risk_table)} test employees.",
        "---"
    ]
    if not monotonic_check:
        stage3_lines.append(
            f"  {cf.WARN_ICON}  {cf.style('[WARNING] Some predictions are not monotonically increasing', cf.YELLOW, cf.BOLD)}"
        )
    else:
        stage3_lines.append(
            f"  {cf.OK_ICON}  {cf.style('[OK] All probabilities are monotonically increasing over time', cf.GREEN, cf.BOLD)}"
        )
        
    print("\n" + cf.make_panel(cf.style("STAGE 3 COMPLETED", cf.GREEN, cf.BOLD), stage3_lines, style_color=cf.GREEN, width=82) + "\n")

    # ================================================================
    #  OUTPUT: VISUALIZATIONS
    # ================================================================
    print(f"\n  {cf.INFO_ICON}  Generating visual analysis charts...")

    plot_pipeline_results(baseline_survival, model, X_test, top_n=12)

    # ================================================================
    #  OUTPUT: REPORTS
    # ================================================================
    print_risk_profiles(risk_table)
    print_executive_report(
        model=model,
        X_test=X_test,
        test_df=test_data,
        X_train=X_train,
        train_df=train_data,
    )

    # Generate HTML Dashboard Report
    generate_xgboost_html_report(
        risk_table=risk_table,
        importance_dict=model.get_booster().get_score(importance_type="gain"),
        train_cindex=train_cindex,
        test_cindex=test_cindex,
        output_path=str(Path(REPORTS_DIR) / "generated" / "report_xgboost.html")
    )




if __name__ == "__main__":
    main()
