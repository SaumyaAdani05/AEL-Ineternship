"""
report.py — Executive Report & Per-Employee Risk Table.

Outputs high-quality CLI dashboards for employee attrition prediction.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline import config
from pipeline import cli_formatter as cf
from pipeline.model import evaluate_model


def print_risk_profiles(
    risk_table: pd.DataFrame,
    max_display: int = None,
) -> None:
    """
    Print a formatted table of per-employee time-horizon risk profiles.

    Highlights employees exceeding the configured risk threshold at any horizon.
    """
    if max_display is None:
        max_display = config.MAX_DISPLAY_EMPLOYEES

    # Initialize terminal virtual process coloring
    cf.init_terminal()

    # Sort by 12-month risk descending (show highest risk first)
    twelve_month_col = list(config.TIME_HORIZONS.keys())[-1]
    sorted_table = risk_table.sort_values(twelve_month_col, ascending=False)

    col_headers = list(risk_table.columns)
    
    # Count high-risk employees across the entire table
    is_high_risk_series = sorted_table.apply(
        lambda r: any(r[col] >= config.HIGH_RISK_THRESHOLD for col in col_headers),
        axis=1
    )
    total_high_risk_count = is_high_risk_series.sum()

    # Prepare table headers and rows
    headers = ["Employee ID"] + col_headers + ["Status"]
    alignments = ["left"] + ["right"] * len(col_headers) + ["center"]
    rows = []

    displayed = 0
    for idx, row in sorted_table.iterrows():
        if displayed >= max_display:
            remaining = len(sorted_table) - max_display
            if remaining > 0:
                # Add a summary row for remaining employees
                dots = cf.style("...", cf.DIM)
                rows.append([dots] + [dots] * len(col_headers) + [dots])
            break

        is_high_risk = is_high_risk_series.loc[idx]
        emp_label = f"EMP_{displayed + 1:04d}"

        # Color employee label and status based on risk
        if is_high_risk:
            styled_emp = cf.style(emp_label, cf.RED, cf.BOLD)
            styled_status = cf.style("● HIGH RISK", cf.RED, cf.BOLD)
        else:
            styled_emp = cf.style(emp_label, cf.GREEN)
            styled_status = cf.style("○ NORMAL", cf.GREEN)

        row_vals = [styled_emp]
        for col in col_headers:
            val = row[col]
            pct_str = f"{val * 100:.1f}%"
            if val >= config.HIGH_RISK_THRESHOLD:
                row_vals.append(cf.style(pct_str, cf.RED, cf.BOLD))
            else:
                row_vals.append(pct_str)
        
        row_vals.append(styled_status)
        rows.append(row_vals)
        displayed += 1

    # Print Table directly with matching width legend
    table_str = cf.make_table(headers, rows, alignments, border_color=cf.GRAY)
    table_width = cf.visual_len(table_str.split('\n')[0])


    # Header title
    title_str = cf.style("EMPLOYEE TIME-HORIZON RISK PROFILES", cf.CYAN, cf.BOLD)
    print("\n" + cf.style(f"✦ {title_str} ✦", cf.CYAN))
    print(table_str)

    # Legend Panel matching width
    threshold_str = cf.style(f"{config.HIGH_RISK_THRESHOLD * 100:.0f}%", cf.YELLOW, cf.BOLD)
    risk_label = cf.style("● HIGH RISK", cf.RED, cf.BOLD)
    
    legend_lines = [
        f"Legend: {risk_label} = Exceeds {threshold_str} attrition threshold at any horizon.",
        f"Total High-Risk Employees Identified: {cf.style(str(total_high_risk_count), cf.RED, cf.BOLD)} / {cf.style(str(len(risk_table)), cf.BLUE)}"
    ]
    panel_width = max(table_width, max(cf.visual_len(l) for l in legend_lines) + 4)
    print(cf.make_panel("", legend_lines, style_color=cf.CYAN, width=panel_width))




def print_executive_report(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    train_df: pd.DataFrame,
) -> None:
    """
    Print an executive summary adapting the existing narrative structure
    (Hidden Killers / False Alarms / Retention Shields) but using XGBoost
    feature importances instead of Cox PH coefficients.
    """
    cf.init_terminal()

    # Get feature importances
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        no_data_title = cf.style("EXECUTIVE SUMMARY: WHAT DRIVES ATTRITION?", cf.RED, cf.BOLD)
        print("\n" + cf.make_panel(no_data_title, ["  No feature importance data available."], style_color=cf.RED))
        return

    imp_series = pd.Series(importance).sort_values(ascending=False)
    max_gain = imp_series.max() if not imp_series.empty else 1.0

    # --- Panel 1: Top Risk Drivers ---
    top_features = imp_series.head(10)
    drivers_lines = [
        "These features represent what the XGBoost model relies on most",
        "to predict employee attrition (ranked by relative information gain).",
        "---"
    ]
    for rank, (feature, gain) in enumerate(top_features.items(), 1):
        clean_name = feature.replace("_", " ")
        if "OverTime" in feature and "Yes" in feature:
            clean_name = "Working Overtime"
        elif "MaritalStatus" in feature:
            clean_name = clean_name.replace("MaritalStatus", "Marital Status:")
        elif "Department" in feature:
            clean_name = clean_name.replace("Department", "Department:")
        elif "Gender" in feature:
            clean_name = clean_name.replace("Gender", "Gender:")
        else:
            clean_name = clean_name.title()

        # Generate styled block progress bar
        # Gradient colors: Red for top 3, Yellow for 4-6, Cyan for others
        if rank <= 3:
            bar_color = cf.RED
        elif rank <= 6:
            bar_color = cf.YELLOW
        else:
            bar_color = cf.CYAN
            
        bar = cf.make_bar(gain, max_gain, length=25, color_code=bar_color)
        rank_str = cf.style(f"{rank:2d}.", cf.BOLD)
        feat_str = f"{clean_name:<26}"
        val_str = cf.style(f"({gain:.0f})", cf.GRAY)
        
        drivers_lines.append(f"  {rank_str} {feat_str} {bar} {val_str}")

    drivers_title = cf.style("TOP KEY RISK DRIVERS", cf.RED, cf.BOLD)
    print("\n" + cf.make_panel(drivers_title, drivers_lines, style_color=cf.RED, width=82))

    # --- Panel 2: Low-Impact Features ---
    low_features = imp_series.tail(5)
    alarm_lines = [
        "These features have minimal predictive power in our survival engine.",
        "Avoid over-investing or designing corporate policies around these:",
        "---"
    ]
    for feature, gain in low_features.items():
        clean_name = feature.replace("_", " ").title()
        if "Overtime" in clean_name:
            clean_name = "Working Overtime"
        alarm_lines.append(f"  {cf.WARN_ICON}  {clean_name:<35} {cf.DIM}Relative Gain: {gain:.1f}{cf.RESET}")

    alarm_title = cf.style("THE FALSE ALARMS", cf.YELLOW, cf.BOLD)
    print("\n" + cf.make_panel(alarm_title, alarm_lines, style_color=cf.YELLOW, width=82))

    # --- Panel 3: Model Accuracy ---
    train_cindex = evaluate_model(model, X_train, train_df)
    test_cindex = evaluate_model(model, X_test, test_df)
    gap = abs(train_cindex - test_cindex)

    accuracy_lines = [
        f"  Training Concordance Index: {cf.style(f'{train_cindex * 100:.1f}%', cf.GREEN, cf.BOLD)}",
        f"  Testing Concordance Index:  {cf.style(f'{test_cindex * 100:.1f}%', cf.GREEN, cf.BOLD)}",
        "---",
        f"  {cf.INFO_ICON}  Interpretation:",
        f"     If we pick two random employees (one who left, one who stayed), the model",
        f"     correctly ranks their attrition risk {cf.style(f'{round(test_cindex * 100)} out of 100', cf.CYAN, cf.BOLD)} times.",
        "     (Scores above 70% represent high enterprise reliability)",
        "---"
    ]

    # Overfitting warning or status
    if gap > 0.05:
        accuracy_lines.append(
            f"  {cf.style('[WARNING]', cf.YELLOW, cf.BOLD)} Generalization Gap: {gap * 100:.1f}%"
        )
        accuracy_lines.append(
            "    Slight overfitting detected. Consider applying stronger regularization."
        )
    else:
        accuracy_lines.append(
            f"  {cf.style('[OK]', cf.GREEN, cf.BOLD)} Generalization Gap: {gap * 100:.1f}% (Healthy model stability)"
        )

    accuracy_title = cf.style("SURVIVAL MODEL GENERALIZATION QUALITY", cf.GREEN, cf.BOLD)
    print("\n" + cf.make_panel(accuracy_title, accuracy_lines, style_color=cf.GREEN, width=82))

    # --- Panel 4: Takeaway Callout ---
    top_feature_name = imp_series.index[0].replace("_", " ").title()
    income_words = ["income", "salary", "rate", "daily", "monthly", "hourly"]
    income_features = [
        f for f in imp_series.index if any(w in f.lower() for w in income_words)
    ]
    top_income_rank = None
    if income_features:
        for rank, feat in enumerate(imp_series.index, 1):
            if feat in income_features:
                top_income_rank = rank
                break

    takeaway_lines = []
    if top_income_rank and top_income_rank <= 3:
        takeaway_lines.append(
            f"  {cf.style('[CRITICAL COMPENSATION ACTION]', cf.RED, cf.BOLD)}"
        )
        takeaway_lines.append(
            "  Pay MATTERS here. Salary/compensation factors reside in the top 3"
        )
        takeaway_lines.append(
            "  attrition drivers. Review compensation bands for high-turnover cohorts."
        )
    elif top_income_rank:
        takeaway_lines.append(
            f"  {cf.style('[RETENTION LEADERSHIP ACTION]', cf.CYAN, cf.BOLD)}"
        )
        takeaway_lines.append(
            f"  Money is a factor (ranked #{top_income_rank}), but NOT the top driver."
        )
        takeaway_lines.append(
            f"  The #1 driver is: {cf.style(top_feature_name, cf.YELLOW, cf.BOLD)}"
        )
        takeaway_lines.append(
            "  Focus leadership training and manager interventions before adjusting salary."
        )
    else:
        takeaway_lines.append(
            f"  {cf.style('[CULTURE & WORK ENVIRONMENT ACTION]', cf.CYAN, cf.BOLD)}"
        )
        takeaway_lines.append(
            f"  The #1 risk driver is: {cf.style(top_feature_name, cf.YELLOW, cf.BOLD)}"
        )
        takeaway_lines.append(
            "  Compensation is not a primary driver. Focus resources on culture,"
        )
        takeaway_lines.append(
            "  work-life balance, and internal role pathways."
        )

    takeaway_title = cf.style("EXECUTIVE DECISION RECOMMENDATION", cf.MAGENTA, cf.BOLD)
    print("\n" + cf.make_panel(takeaway_title, takeaway_lines, style_color=cf.MAGENTA, width=82) + "\n")

