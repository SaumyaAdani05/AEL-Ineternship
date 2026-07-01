"""
visualization.py — Single interactive visualization app for the XGBoost Survival Pipeline.

Combines three key views in a single window with Page Up/Down and Previous/Next controls:
  - View 1: Key Attrition Drivers (Feature Importance)
  - View 2: Individual Time-Shifts (Baseline vs Shifted Curves)
  - View 3: Cohort Retention Comparison (High-Risk vs Low-Risk Groups)
"""
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# pyrefly: ignore [missing-import]
import xgboost as xgb

from pipeline import config
from pipeline.model import (
    compute_individual_survival_curve,
    predict_risk_multipliers,
    predict_risk_scores,
)

# Consistent style
sns.set_style("whitegrid")
COLORS = {
    "high_risk": "#E74C3C",
    "low_risk": "#2ECC71",
    "baseline": "#3498DB",
    "accent": "#F39C12",
    "neutral": "#95A5A6",
    "bg_light": "#F8F9F9",
    "text_dark": "#2C3E50",
}


def plot_pipeline_results(
    baseline_survival: pd.Series,
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    top_n: int = 12,
) -> None:
    """
    Renders an interactive, unified 3-page visualization dashboard in a single window.
    Supports navigation via keyboard arrow keys (left/right) and GUI buttons.
    """
    from matplotlib.widgets import Button

    # Setup the figure
    fig = plt.figure(figsize=(12, 7.5))
    
    # Adjust spacing to keep a clean margin at the bottom for navigation buttons
    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.15, right=0.93)

    # Add subplot axes stacked on top of each other
    ax1 = fig.add_subplot(111)  # Page 1: Feature Importance
    ax2 = fig.add_subplot(111)  # Page 2: Shifted Curves
    ax3 = fig.add_subplot(111)  # Page 3: Cohort Comparison

    # =========================================================================
    #  PAGE 1: FEATURE IMPORTANCE (KEY DRIVERS)
    # =========================================================================
    importance = model.get_booster().get_score(importance_type="gain")
    if importance:
        imp_series = pd.Series(importance).sort_values(ascending=False).head(top_n)
        
        # Format names for human readability
        clean_names = []
        for name in imp_series.index:
            n = name.replace("_", " ").replace("  ", " ").title()
            n = (n.replace("Overtime Yes", "Working Overtime")
                  .replace("Maritalstatus Single", "Marital Status: Single")
                  .replace("Department Sales", "Department: Sales")
                  .replace("Department Research & Development", "Department: R&D")
                  .replace("Gender Male", "Gender: Male")
                  .replace("Maritalstatus Married", "Marital Status: Married")
                  .replace("Yearswithcurrmanager", "Years With Current Manager")
                  .replace("Yearsincurrentrole", "Years In Current Role")
                  .replace("Totalworkingyears", "Total Working Years")
                  .replace("Monthlyincome", "Monthly Income")
                  .replace("Educationfield", "Education Field")
                  .replace("Stockoptionlevel", "Stock Option Level")
                  .replace("Percentsalaryhike", "Percent Salary Hike")
                  .replace("Performancerating", "Performance Rating")
                  .replace("Jobinvolvement", "Job Involvement"))
            clean_names.append(n)

        bars = ax1.barh(
            range(len(imp_series)),
            imp_series.values,
            color=sns.color_palette("viridis", len(imp_series)),
            edgecolor="white",
            linewidth=0.5,
        )
        ax1.set_yticks(range(len(imp_series)))
        ax1.set_yticklabels(clean_names, fontsize=10, fontweight='bold', color='#34495E')
        ax1.invert_yaxis()
        ax1.set_xlabel("Importance Score (Gain)", fontsize=11, fontweight="bold", color='#2C3E50')
        ax1.set_title("Which Employee Attributes Drive Flight Risk Most?", fontsize=13, fontweight="bold", pad=12, color=COLORS["text_dark"])
        
        # Add labels to the ends of the bars
        for bar_item, val in zip(bars, imp_series.values):
            ax1.text(
                val + max(imp_series.values) * 0.01,
                bar_item.get_y() + bar_item.get_height() / 2,
                f"{val:.1f}",
                va="center",
                fontsize=9,
                color="#333",
                fontweight='bold',
            )
            
        # Explanatory text box
        desc_text = (
            "XGBoost analyzed all employee features to determine which ones carry the most predictive weight.\n"
            "Higher scores represent features that have the strongest influence on when an employee is likely to resign.\n"
            "Interventions focusing on the top 3 attributes will yield the highest retention returns."
        )
        ax1.text(
            0.5, -0.15, desc_text, transform=ax1.transAxes,
            ha="center", va="top", fontsize=9.5, color="#5D6D7E",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["bg_light"], edgecolor="#BDC3C7", alpha=0.8)
        )
    else:
        ax1.text(0.5, 0.5, "No feature importance data available", ha="center", va="center")

    # =========================================================================
    #  PAGE 2: INDIVIDUAL SHIFTED SURVIVAL CURVES (TIME-SHIFT)
    # =========================================================================
    risk_scores = predict_risk_scores(model, X_test)
    sorted_indices = np.argsort(risk_scores)
    n = len(sorted_indices)

    # Representative percentile samples to show a clear spectrum of risk
    selected_positions = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    labels = ["Lowest Flight Risk", "Low-Medium Risk", "Median Flight Risk", "High-Medium Risk", "Highest Flight Risk"]
    colors = ["#2ECC71", "#27AE60", "#F39C12", "#E67E22", "#E74C3C"]

    max_time = min(baseline_survival.index.max(), 5.0)  # cap visualization at 5 years
    time_grid = np.linspace(0, max_time, 200)

    # Plot baseline S0(t)
    baseline_interp = np.interp(time_grid, baseline_survival.index.values, baseline_survival.values, left=1.0)
    ax2.plot(
        time_grid,
        baseline_interp * 100,
        color=COLORS["baseline"],
        linewidth=3.5,
        linestyle="--",
        label="Company Average Baseline",
        alpha=0.9,
    )

    # Plot representative curves
    for i, pos in enumerate(selected_positions):
        idx = sorted_indices[pos]
        score = risk_scores[idx]
        multiplier = np.exp(np.clip(score, -20.0, 20.0))
        
        curve = compute_individual_survival_curve(baseline_survival, score, time_grid)
        lbl = labels[i]
        ax2.plot(
            time_grid,
            curve.values * 100,
            color=colors[i],
            linewidth=2.2,
            label=f"{lbl} ({multiplier:.1f}x risk multiplier)",
            alpha=0.9,
        )

    # Reference lines and timeline ticks
    ax2.axhline(y=50, color="#95A5A6", linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.text(max_time * 0.02, 52, "50% Retention Probability Threshold", fontsize=9, color="#7F8C8D", fontweight='bold')

    for label, t in config.TIME_HORIZONS.items():
        if t <= max_time:
            ax2.axvline(x=t, color="#BDC3C7", linestyle=":", linewidth=1, alpha=0.5)
            ax2.text(t + 0.02, 8, label, fontsize=8, color="#7F8C8D", rotation=90, fontweight="bold")

    ax2.set_title("Time-Shift Mechanism: Baseline vs Individual Survival Curves", fontsize=13, fontweight="bold", pad=12, color=COLORS["text_dark"])
    ax2.set_xlabel("Years at Company", fontsize=11, fontweight="bold", color='#2C3E50')
    ax2.set_ylabel("Retention Probability (%)", fontsize=11, fontweight="bold", color='#2C3E50')
    ax2.set_ylim(0, 105)
    ax2.set_xlim(0, max_time)
    ax2.legend(loc="lower left", fontsize=9.5, framealpha=0.9)

    # Explanatory text box
    desc_text_2 = (
        "Each line represents the predicted retention probability for a specific employee over time.\n"
        "The Baseline represents the average company survival timeline. Proportional hazard multipliers shift the curves:\n"
        "High-risk employees shift left (retention drops rapidly), while low-risk employees stretch to the right."
    )
    ax2.text(
        0.5, -0.15, desc_text_2, transform=ax2.transAxes,
        ha="center", va="top", fontsize=9.5, color="#5D6D7E",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["bg_light"], edgecolor="#BDC3C7", alpha=0.8)
    )

    # =========================================================================
    #  PAGE 3: COHORT ATTRITION CURVES (HIGH vs LOW RISK GROUPS)
    # =========================================================================
    risk_multipliers = predict_risk_multipliers(model, X_test)
    median_multiplier = np.median(risk_multipliers)

    high_risk_mask = risk_multipliers >= median_multiplier
    low_risk_mask = ~high_risk_mask

    def average_cohort_curve(mask: np.ndarray) -> np.ndarray:
        scores = risk_scores[mask]
        curves = np.zeros((len(scores), len(time_grid)))
        for j, scr in enumerate(scores):
            curve = compute_individual_survival_curve(baseline_survival, scr, time_grid)
            curves[j] = curve.values
        return curves.mean(axis=0) if len(curves) > 0 else np.ones(len(time_grid))

    high_risk_avg = average_cohort_curve(high_risk_mask)
    low_risk_avg = average_cohort_curve(low_risk_mask)

    n_high = high_risk_mask.sum()
    n_low = low_risk_mask.sum()

    ax3.plot(
        time_grid,
        high_risk_avg * 100,
        color=COLORS["high_risk"],
        linewidth=2.8,
        label=f"High-Risk Group (n={n_high}) - Above Median Risk",
    )
    ax3.plot(
        time_grid,
        low_risk_avg * 100,
        color=COLORS["low_risk"],
        linewidth=2.8,
        label=f"Low-Risk Group (n={n_low}) - Below Median Risk",
    )

    # Shade the risk gap
    ax3.fill_between(
        time_grid,
        high_risk_avg * 100,
        low_risk_avg * 100,
        alpha=0.15,
        color=COLORS["high_risk"],
        label="Flight Risk Gap (Retention Window)",
    )

    six_month = 0.5
    ax3.axvline(x=six_month, color=COLORS["baseline"], linestyle="--", linewidth=1.5)
    ax3.text(
        six_month + 0.03, 45, "Critical 6-Month Mark", fontsize=10,
        color=COLORS["baseline"], fontweight="bold",
    )
    ax3.axhline(y=50, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax3.set_title("Retention Gap: High-Risk vs Low-Risk Employee Cohorts", fontsize=13, fontweight="bold", pad=12, color=COLORS["text_dark"])
    ax3.set_xlabel("Years at Company", fontsize=11, fontweight="bold", color='#2C3E50')
    ax3.set_ylabel("Group Average Retention Probability (%)", fontsize=11, fontweight="bold", color='#2C3E50')
    ax3.set_ylim(0, 105)
    ax3.set_xlim(0, max_time)
    ax3.legend(loc="lower left", fontsize=10)

    # Explanatory text box
    desc_text_3 = (
        "Comparing the average retention timelines of employees above vs. below the median flight risk.\n"
        "The shaded area represents the Risk Gap. It highlights the potential window where active management\n"
        "and custom retention strategies can prevent premature attrition, especially before the critical 6-month mark."
    )
    ax3.text(
        0.5, -0.15, desc_text_3, transform=ax3.transAxes,
        ha="center", va="top", fontsize=9.5, color="#5D6D7E",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["bg_light"], edgecolor="#BDC3C7", alpha=0.8)
    )

    # =========================================================================
    #  INTERACTIVE CONTROLS
    # =========================================================================
    current_view = 0

    def show_view(view_index: int) -> None:
        nonlocal current_view
        current_view = view_index
        if current_view == 0:
            ax1.set_visible(True)
            ax2.set_visible(False)
            ax3.set_visible(False)
            fig.suptitle("Attrition Risk Dashboard (View 1 of 3: Key Attrition Drivers)", fontsize=13, fontweight='bold', color='#2C3E50')
        elif current_view == 1:
            ax1.set_visible(False)
            ax2.set_visible(True)
            ax3.set_visible(False)
            fig.suptitle("Attrition Risk Dashboard (View 2 of 3: Individual Time-Shifts)", fontsize=13, fontweight='bold', color='#2C3E50')
        else:
            ax1.set_visible(False)
            ax2.set_visible(False)
            ax3.set_visible(True)
            fig.suptitle("Attrition Risk Dashboard (View 3 of 3: Cohort Retention Comparison)", fontsize=13, fontweight='bold', color='#2C3E50')
        fig.canvas.draw_idle()

    # Keyboard navigation listener
    def on_key(event) -> None:
        if event.key == "left":
            show_view((current_view - 1) % 3)
            update_toolbar_button_states()
        elif event.key == "right":
            show_view((current_view + 1) % 3)
            update_toolbar_button_states()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Add navigation buttons to the bottom of the figure
    ax_prev = fig.add_axes([0.72, 0.03, 0.1, 0.045])
    ax_next = fig.add_axes([0.84, 0.03, 0.1, 0.045])

    btn_prev = Button(ax_prev, "Previous", color="#BDC3C7", hovercolor="#95A5A6")
    btn_next = Button(ax_next, "Next", color="#3498DB", hovercolor="#2980B9")

    # Set button label font sizes
    btn_prev.label.set_fontsize(9)
    btn_prev.label.set_fontweight('bold')
    btn_next.label.set_fontsize(9)
    btn_next.label.set_fontweight('bold')

    def click_prev(event=None) -> None:
        show_view((current_view - 1) % 3)
        update_toolbar_button_states()

    def click_next(event=None) -> None:
        show_view((current_view + 1) % 3)
        update_toolbar_button_states()

    btn_prev.on_clicked(click_prev)
    btn_next.on_clicked(click_next)

    # Configure native toolbar back/forward buttons (if using TkAgg or compatible backend)
    toolbar = fig.canvas.manager.toolbar

    def update_toolbar_button_states() -> None:
        if toolbar is not None:
            try:
                toolbar._buttons["Back"].config(state="normal")
                toolbar._buttons["Forward"].config(state="normal")
            except Exception:
                pass

    if toolbar is not None:
        try:
            toolbar._buttons["Back"].config(command=click_prev)
            toolbar._buttons["Forward"].config(command=click_next)
            toolbar.set_history_buttons = update_toolbar_button_states
        except Exception:
            pass

    # Show initial page
    show_view(0)
    update_toolbar_button_states()
    plt.show()
