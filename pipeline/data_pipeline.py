"""
data_pipeline.py — Stage 1: The Unified Data Distribution.

Implements the encoding strategy from the Architectural Integration PDF:
  1. Load raw CSV and isolate clean rows
  2. Ordinal encoding (preserves rank)
  3. One-hot encoding (isolates nominal states)
  4. Leave-One-Out encoding (compresses high-cardinality to risk %)
  5. Min-Max scaling (normalizes continuous numbers to 0.0–1.0)

Output: A pristine, 100% numerical matrix — the "Single Source of Truth".
"""
import warnings
from typing import Tuple

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pipeline import config
from pipeline import cli_formatter as cf

warnings.filterwarnings("ignore", category=FutureWarning)



def load_dataset(path: str) -> pd.DataFrame:
    """
    Load CSV and auto-detect the format boundary.

    The datasets.csv file has a format change around row 1490 where the second
    half switches from text labels to raw numeric codes with extra columns.
    This function loads only the clean labeled rows.
    """
    # Read with flexible error handling to detect the boundary
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        # Older pandas versions use error_bad_lines
        df = pd.read_csv(path, error_bad_lines=False)

    print(f"  {cf.OK_ICON} Loaded {len(df)} rows from '{path}'")
    return df


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ordinal mappings, cast numeric types, and handle edge cases.

    Steps:
      - Map ordinal text labels to integer ranks
      - Convert Attrition to binary (True/False → 1/0)
      - Cast remaining columns to numeric
      - Fix YearsAtCompany == 0 for attrited employees (set to 0.01)
    """
    data = data.copy()

    # Map ordinal columns using config mappings
    for col, mapping in config.ORDINAL_MAPPINGS.items():
        if col in data.columns:
            data[col] = data[col].replace(mapping)

    # Convert Attrition to binary integer
    if data[config.EVENT_COL].dtype == object:
        data[config.EVENT_COL] = (data[config.EVENT_COL] == "Yes").astype(int)
    else:
        data[config.EVENT_COL] = data[config.EVENT_COL].astype(int)

    # Cast non-text columns to numeric
    for col in data.columns:
        if col not in config.SKIP_NUMERIC_CAST:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Fix zero-duration edge case: attrited employees with 0 years
    # (would cause math errors in hazard calculations)
    zero_dur_attrited = (data[config.DURATION_COL] == 0) & (
        data[config.EVENT_COL] == 1
    )
    data.loc[zero_dur_attrited, config.DURATION_COL] = 0.01

    data = data.dropna()
    print(f"  {cf.OK_ICON} Preprocessed: {len(data)} clean rows")
    return data


def split_data(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split preserving attrition ratio.
    """
    train_data, test_data = train_test_split(
        data,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=data[config.EVENT_COL],
    )
    print(f"  {cf.OK_ICON} Train: {len(train_data)} rows | Test: {len(test_data)} rows")
    return train_data, test_data


def encode_categoricals(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, ce.LeaveOneOutEncoder]:
    """
    Apply one-hot and Leave-One-Out encoding to categorical columns.

    One-hot: Gender, MaritalStatus, OverTime, Department (low-cardinality)
    LOO: JobRole, EducationField (high-cardinality → compressed to risk %)

    Returns encoded train, test, and the fitted LOO encoder for reuse.
    """
    train_data = train_data.copy()
    test_data = test_data.copy()

    # --- One-hot encoding ---
    existing_ohe_cols = [c for c in config.ONE_HOT_COLS if c in train_data.columns]
    if existing_ohe_cols:
        train_data = pd.get_dummies(
            train_data, columns=existing_ohe_cols, drop_first=True, dtype=int
        )
        test_data = pd.get_dummies(
            test_data, columns=existing_ohe_cols, drop_first=True, dtype=int
        )

        # Align columns (train may create dummies test doesn't have, and vice versa)
        train_data, test_data = train_data.align(test_data, join="left", axis=1)
        test_data = test_data.fillna(0)

    # --- Leave-One-Out encoding ---
    existing_loo_cols = [c for c in config.LOO_COLS if c in train_data.columns]
    loo_encoder = ce.LeaveOneOutEncoder(cols=existing_loo_cols)

    if existing_loo_cols:
        train_data = loo_encoder.fit_transform(
            train_data, train_data[config.EVENT_COL]
        )
        test_data = loo_encoder.transform(test_data)

    train_data = train_data.dropna()
    test_data = test_data.dropna()

    print(
        f"  {cf.OK_ICON} Encoded: {len(train_data.columns)} features "
        f"(OHE: {len(existing_ohe_cols)}, LOO: {len(existing_loo_cols)})"
    )
    return train_data, test_data, loo_encoder


def scale_continuous(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Apply Min-Max scaling (0.0–1.0) to all continuous numeric columns.

    Fitted on training data only to prevent data leakage.
    Survival columns (YearsAtCompany, Attrition) are excluded from scaling.

    Returns scaled train, test, and the fitted scaler for inverse transforms.
    """
    train_data = train_data.copy()
    test_data = test_data.copy()

    # Identify continuous columns (everything numeric except survival targets)
    exclude_cols = {config.DURATION_COL, config.EVENT_COL}
    continuous_cols = [
        col
        for col in train_data.columns
        if col not in exclude_cols and train_data[col].dtype in [np.float64, np.int64, np.int32, np.float32, int, float]
    ]

    scaler = MinMaxScaler()
    train_data[continuous_cols] = scaler.fit_transform(train_data[continuous_cols])
    test_data[continuous_cols] = scaler.transform(test_data[continuous_cols])

    print(f"  {cf.OK_ICON} Scaled {len(continuous_cols)} continuous features to [0, 1]")
    return train_data, test_data, scaler

def build_unified_distribution(
    dataset_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, ce.LeaveOneOutEncoder]:
    cf.init_terminal()
    title = cf.style("STAGE 1: BUILDING THE UNIFIED DATA DISTRIBUTION", cf.CYAN, cf.BOLD)

    print("\n" + cf.make_panel(title, [
        "Initializing Unified Data Distribution Pipeline...",
        f"Target Dataset Path: {dataset_path}"
    ], style_color=cf.CYAN, width=82))

    raw_data = load_dataset(dataset_path)
    preprocessed = preprocess_data(raw_data)
    train_data, test_data = split_data(preprocessed)
    train_data, test_data, loo_encoder = encode_categoricals(train_data, test_data)
    train_data, test_data, scaler = scale_continuous(train_data, test_data)

    summary_lines = [
        "Unified Distribution ready: Pristine 100% numerical matrix built.",
        "---",
        f"  {cf.OK_ICON}  Features Encoded: {cf.style(str(len(train_data.columns)), cf.GREEN, cf.BOLD)} columns",
        f"  {cf.INFO_ICON}  Train Set Shape: {cf.style(str(train_data.shape), cf.GRAY)}",
        f"  {cf.INFO_ICON}  Test Set Shape:  {cf.style(str(test_data.shape), cf.GRAY)}"
    ]
    print("\n" + cf.make_panel(cf.style("STAGE 1 COMPLETED", cf.GREEN, cf.BOLD), summary_lines, style_color=cf.GREEN, width=82) + "\n")

    return train_data, test_data, scaler, loo_encoder

