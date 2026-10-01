# Insurance Cost Regression

A reproducible regression project that predicts medical insurance charges. It replaces notebook-wide mutation with a train/test-safe scikit-learn pipeline.

## What changed from the notebook

- BMI and charges retain decimal precision.
- Category encoding, imputation, and scaling are fitted only on the training set through a pipeline.
- No feature is selected from a target-derived bin created from all rows.
- A mean baseline, Ridge regression, and random forest are compared with MAE, RMSE, and R².

## Results

On the supplied 1,338-row CSV, the selected Random Forest achieved **MAE $2,413.91**, **RMSE $4,388.55**, and **R² 0.8759** on 268 held-out rows using an 80/20 split (`random_state=42`). This is a dataset estimate only; it must not be used to make insurance or medical decisions.

## Run

```powershell
python -m venv .venv  # Create an isolated environment.
.\.venv\Scripts\python -m pip install -r requirements.txt  # Install dependencies.
.\.venv\Scripts\python train.py --data "C:\Users\lenovo\Downloads\ML\insurance.csv"  # Train and evaluate the project.
```

The best pipeline and its evaluation report are written to `artifacts/`.
