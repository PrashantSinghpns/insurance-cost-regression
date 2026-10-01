"""Train and evaluate an insurance-charge regression pipeline without data leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluation import evaluate_and_export
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
REQUIRED_COLUMNS = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET]


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate numeric inputs and make categorical missing values sklearn-compatible."""
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column]).any():
            raise ValueError(f"Feature {column} contains an infinite value.")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        frame[column] = frame[column].replace("", np.nan).astype(object)
        frame[column] = frame[column].where(frame[column].notna(), np.nan)
    return frame


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load the data and apply only defensible schema and target checks."""
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")
    if np.isinf(frame[TARGET]).any():
        raise ValueError("Targets must be finite.")
    frame = frame.dropna(subset=[TARGET])
    frame = frame.loc[frame[TARGET] > 0].copy()
    return normalize_features(frame)


def make_preprocessor() -> ColumnTransformer:
    """Return transformations that are trained inside each model pipeline."""
    numeric_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    category_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_steps, NUMERIC_FEATURES), ("categorical", category_steps, CATEGORICAL_FEATURES)]
    )



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the documented CSV schema")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    frame = load_and_clean(args.data)
    candidates = {"dummy_mean": DummyRegressor(strategy="mean"), "ridge": Ridge(alpha=10.0),
                  "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                        random_state=42, n_jobs=-1)}
    report = evaluate_and_export(frame, TARGET, make_preprocessor, candidates, args.data,
                                 args.output_dir, classification=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
