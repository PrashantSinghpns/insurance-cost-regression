"""Train and evaluate an insurance-charge regression pipeline without data leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "charges"  # Medical insurance charges are the regression target.
NUMERIC_FEATURES = ["age", "bmi", "children"]  # Preserve BMI and charges as floats rather than truncating them.
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]  # Encode categories without assigning artificial numeric order.
REQUIRED_COLUMNS = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET]  # Keep input validation explicit.


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load the data and apply only defensible schema and target checks."""
    frame = pd.read_csv(path)  # Load a local authorised copy of the dataset.
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))  # Identify incompatible CSVs immediately.
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")  # Stop before an unclear downstream failure.
    frame = frame[REQUIRED_COLUMNS].copy()  # Exclude accidental extra columns from the model.
    frame = frame.dropna(subset=[TARGET])  # A missing charge cannot be a supervised target.
    frame = frame.loc[frame[TARGET] > 0].copy()  # Charges must be positive for this task.
    return frame


def make_preprocessor() -> ColumnTransformer:
    """Return transformations that are trained inside each model pipeline."""
    numeric_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Learn replacements from training rows only.
            ("scaler", StandardScaler()),  # Standardise continuous variables for the ridge baseline.
        ]
    )
    category_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Preserve rows with a missing category.
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),  # Handle unseen future regions safely.
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_steps, NUMERIC_FEATURES), ("categorical", category_steps, CATEGORICAL_FEATURES)]  # Keep transformations column-specific.
    )


def metrics(y_true: pd.Series, predicted) -> dict[str, float]:
    """Calculate held-out errors in the original currency scale."""
    return {
        "mae": round(float(mean_absolute_error(y_true, predicted)), 2),  # Typical absolute charge error.
        "rmse": round(float(mean_squared_error(y_true, predicted) ** 0.5), 2),  # Penalise large errors more strongly.
        "r2": round(float(r2_score(y_true, predicted)), 4),  # Explainability of unseen charge variation.
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)  # Expose a reproducible entry point.
    parser.add_argument("--data", required=True, help="Path to insurance.csv")  # Keep source data separate from source code.
    parser.add_argument("--output-dir", default="artifacts", help="Directory for trained artifacts")  # Group output files predictably.
    args = parser.parse_args()

    frame = load_and_clean(args.data)  # Clean before the train/test split without using target-derived features.
    X = frame.drop(columns=TARGET)  # Ensure charges never enter the feature pipeline.
    y = frame[TARGET]  # Keep target labels separate.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # Hold out 20% of data for final comparison.
    )
    candidates = {
        "dummy_mean": DummyRegressor(strategy="mean"),  # Establish a no-skill reference.
        "ridge": Ridge(alpha=10.0),  # Offer an interpretable regularised baseline.
        "random_forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, random_state=42, n_jobs=-1  # Learn non-linear smoker and BMI effects.
        ),
    }
    results: dict[str, dict[str, float]] = {}  # Save every candidate's test metrics.
    trained = {}  # Retain fitted pipelines for model selection.
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])  # Keep every learned step together.
        pipeline.fit(X_train, y_train)  # Fit exclusively on the training partition.
        results[name] = metrics(y_test, pipeline.predict(X_test))  # Evaluate only on untouched rows.
        trained[name] = pipeline  # Store the candidate pipeline for export.

    best_name = min(results, key=lambda name: results[name]["rmse"])  # Prefer the smallest held-out error.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the artifact folder when needed.
    joblib.dump(trained[best_name], output_dir / "model.joblib")  # Persist preprocessing and estimator together.
    report = {"dataset_rows": len(frame), "train_rows": len(X_train), "test_rows": len(X_test), "best_model": best_name, "test_metrics": results}  # Capture reproducibility facts.
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")  # Save a reviewable evaluation report.
    print(json.dumps(report, indent=2))  # Show the report after training.


if __name__ == "__main__":
    main()  # Execute training only for direct script runs.
