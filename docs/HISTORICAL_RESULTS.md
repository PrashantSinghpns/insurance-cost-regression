# Historical evaluation record

The text below preserves the prior README result for traceability. Its source CSV and saved training artifacts are not included here, so these numbers were not reproduced in this review.

The old regression workflow selected its winner using test RMSE. Those reported scores are selection-biased and do not validate the revised training-only cross-validation workflow.

## Results

On the supplied 1,338-row CSV, the selected Random Forest achieved **MAE $2,413.91**, **RMSE $4,388.55**, and **R² 0.8759** on 268 held-out rows using an 80/20 split (`random_state=42`). This is a dataset estimate only; it must not be used to make insurance or medical decisions.
