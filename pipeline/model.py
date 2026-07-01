"""
model.py — Stage 2 + 3: XGBoost Survival Engine & Time-Shift Mechanism.

Stage 2 (Convergence):
  - Trains XGBoost with `survival:cox` objective
  - Outputs log-hazard ratios (risk scores) per employee
  - Exponentiates to "Risk Multipliers" (e.g., 3.4× baseline)

Stage 3 (Time-Shift):
  - Estimates baseline survival curve via Nelson-Aalen estimator
  - Shifts baseline by individual risk multipliers: S(t) = S₀(t)^exp(score)
  - Queries shifted curves at 1M, 3M, 6M, 12M → time-bound probabilities
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index

from pipeline import config


# ============================================================================
#  STAGE 2: XGBOOST SURVIVAL ENGINE
# ============================================================================


def prepare_survival_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare features (X) and survival target (y) for XGBoost survival:cox.

    The survival:cox objective expects:
      - Positive values for uncensored (event observed, i.e. attrited)
      - Negative values for censored (still employed)

    The absolute value is the duration (YearsAtCompany).

    Returns:
        X: Feature matrix (all columns except duration and event)
        y: Signed duration array (positive = attrited, negative = censored)
    """
    exclude_cols = [config.DURATION_COL, config.EVENT_COL]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()

    duration = df[config.DURATION_COL].values.astype(np.float64)
    event = df[config.EVENT_COL].values.astype(np.int32)

    # survival:cox convention: positive = event, negative = censored
    y = np.where(event == 1, duration, -duration)

    return X, y


def train_xgboost_survival(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
) -> xgb.XGBRegressor:
    """
    Train an XGBoost model with survival:cox objective.

    The model learns to predict log-hazard ratios that rank employees
    by their relative risk of attrition, similar to Cox PH coefficients
    but with the ability to capture non-linear interactions.

    If validation data is provided, early stopping is applied.
    Otherwise, trains for the full n_estimators.
    """
    params = config.XGBOOST_PARAMS.copy()

    # Add early stopping config if validation data will be provided
    if X_val is not None and y_val is not None:
        params["early_stopping_rounds"] = config.EARLY_STOPPING_ROUNDS

    model = xgb.XGBRegressor(**params)

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)

    return model


def predict_risk_scores(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    Predict log-hazard ratios (risk scores) for each employee.

    Raw output from survival:cox with output_margin=True is the log partial hazard.
    """
    return model.predict(X, output_margin=True)


def predict_risk_multipliers(
    model: xgb.XGBRegressor, X: pd.DataFrame
) -> np.ndarray:
    """
    Convert log-hazard ratios to risk multipliers.

    A multiplier of 3.4 means "this employee is 3.4x more likely to
    experience the attrition event than the baseline employee."
    """
    # model.predict(X) already returns the hazard ratio exp(margin) by default
    risk_multipliers = model.predict(X)
    return np.clip(risk_multipliers, 0.0001, 10000.0)


def evaluate_model(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    df: pd.DataFrame,
) -> float:
    """
    Compute the concordance index (C-index) on a dataset.

    C-index measures the model's ability to correctly rank employees
    by risk. Values > 0.70 are considered good.
    """
    risk_scores = predict_risk_scores(model, X)
    durations = df[config.DURATION_COL].values
    events = df[config.EVENT_COL].values

    c_index = concordance_index(
        event_times=durations,
        predicted_scores=-risk_scores,  # negate: higher risk = shorter survival
        event_observed=events,
    )
    return c_index


# ============================================================================
#  STAGE 3: TIME-SHIFT MECHANISM
# ============================================================================


def estimate_baseline_survival(
    train_df: pd.DataFrame,
) -> Tuple[NelsonAalenFitter, pd.Series]:
    """
    Estimate the baseline cumulative hazard and survival curve using
    the Nelson-Aalen estimator on training data.

    The baseline survival curve S₀(t) represents the "average" employee's
    survival trajectory — the default timeline before individual risk
    adjustments.

    Returns:
        naf: Fitted NelsonAalenFitter (contains cumulative hazard)
        baseline_survival: S₀(t) as a pandas Series indexed by time
    """
    naf = NelsonAalenFitter()
    naf.fit(
        durations=train_df[config.DURATION_COL],
        event_observed=train_df[config.EVENT_COL],
    )

    # Baseline survival: S₀(t) = exp(-H₀(t))
    cumulative_hazard = naf.cumulative_hazard_.squeeze()
    baseline_survival = np.exp(-cumulative_hazard)
    baseline_survival.name = "baseline_survival"

    return naf, baseline_survival


def compute_individual_survival_curve(
    baseline_survival: pd.Series,
    risk_score: float,
    time_points: np.ndarray = None,
) -> pd.Series:
    """
    Shift the baseline survival curve by an individual's risk score.

    Uses the proportional hazards formula:
        S(t|x) = S₀(t) ^ exp(risk_score)

    Where:
      - S₀(t) is the baseline survival from Nelson-Aalen
      - risk_score is the log-hazard ratio from XGBoost
      - exp(risk_score) is the "Risk Multiplier"

    A higher risk_score compresses the curve downward (faster attrition).
    A lower/negative risk_score stretches it upward (slower attrition).
    """
    # Clamp risk_score to prevent numerical overflow in exp()
    risk_score_clamped = np.clip(risk_score, -20.0, 20.0)
    multiplier = np.exp(risk_score_clamped)

    if time_points is not None:
        # Interpolate baseline survival to requested time points
        baseline_interp = np.interp(
            time_points,
            baseline_survival.index.values,
            baseline_survival.values,
            left=1.0,  # survival = 1.0 at t=0
            right=baseline_survival.values[-1],
        )
        individual_survival = np.power(baseline_interp, multiplier)
        return pd.Series(individual_survival, index=time_points, name="survival")
    else:
        individual_survival = np.power(baseline_survival.values, multiplier)
        return pd.Series(
            individual_survival,
            index=baseline_survival.index,
            name="survival",
        )


def predict_time_horizon_probabilities(
    model: xgb.XGBRegressor,
    baseline_survival: pd.Series,
    X: pd.DataFrame,
    horizons: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Compute per-employee attrition probabilities at specific time horizons.

    This is the final business output: for each employee, the probability
    of leaving within 1 month, 3 months, 6 months, and 12 months.

    Probability of attrition = 1 - S(t|x)

    Returns a DataFrame with columns like:
        1-Month | 3-Month | 6-Month | 12-Month
         0.021  |  0.053  |  0.452  |   0.824
    """
    if horizons is None:
        horizons = config.TIME_HORIZONS

    risk_scores = predict_risk_scores(model, X)
    time_points = np.array(list(horizons.values()))

    results = []
    for score in risk_scores:
        survival_at_horizons = compute_individual_survival_curve(
            baseline_survival, score, time_points
        )
        # Attrition probability = 1 - survival probability
        attrition_probs = 1.0 - survival_at_horizons.values
        # Clamp to [0, 1]
        attrition_probs = np.clip(attrition_probs, 0.0, 1.0)
        results.append(attrition_probs)

    result_df = pd.DataFrame(
        results,
        columns=list(horizons.keys()),
        index=X.index,
    )
    return result_df
