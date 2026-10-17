# Stories Branch-Product Predictor

This project now contains only the statistical predictor layer. It does not
apply business thresholds, rank issues, recommend actions, or maintain decision
statuses. Its versioned output is intended to be consumed by a separate decision
model later.

The prompt originally names one `stories_transactions.csv`; the supplied data
is stored as chronological `data/train.csv`, `data/validation.csv`, and
`data/test.csv`. The default pipeline enforces these roles:

1. Fit Huber and OLS two-way fixed-effects models on `train.csv` only.
2. Score `validation.csv` without refitting and inspect effect stability.
3. Freeze the predictor protocol.
4. Score `test.csv` without refitting.
5. Export decision-neutral branch-product predictions and residuals.

## Run

From the repository root:

```powershell
python -m pip install -r branch-analysis-model/requirements.txt
python branch-analysis-model/main.py
```

## Predictor contract

The primary downstream artifacts are:

- `outputs/predictor_output.json`: versioned JSON payload for a future decision model.
- `outputs/decision_model_input.json`: compact nested evidence contract intended for the downstream decision model.
- `outputs/predictor_output.csv`: the same 135 branch-product records in tabular form.
- `outputs/trained_models.json`: training-fitted Huber and OLS parameters and residuals.
- `outputs/holdout_predictions.csv`: validation and test predictions for both methods.
- `outputs/holdout_evaluation.json`: leakage checks and validation/test metrics.
- `outputs/predictor_report.html`: technical model diagnostics only.

Each predictor record contains financial context plus:

- `predicted_unit_cost`
- `predicted_listed_unit_price`
- `cost_log_residual` and `cost_residual_pct`
- `price_log_residual` and `price_residual_pct`
- absolute and percentage errors
- `prediction_complete`, indicating that both categorical levels were observed in training

It intentionally contains no recommendation, action, threshold, priority score,
or management status.

`decision_model_input.json` groups each record into `entity`, `observation`,
`model_expectation`, `residual_signals`, and `prediction_quality`. Stable
`evidence_id` values allow a later decision model to track the same
branch-product evidence across runs.

## Interpretation

The model is a two-way branch/product fixed-effects predictor:

```text
log(value_branch,product) = product effect + branch effect + residual
```

Huber iteratively reweighted least squares is the primary robust method; OLS is
retained as a comparison. Models train on one distinct observation per
branch-product combination, rather than treating repeated transaction lines as
independent structural evidence.

The current dataset keeps cost and listed price constant within each
branch-product combination. Consequently, this predictor estimates structural
expected values and residuals; it is not a demand, sales-volume, or causal profit
predictor.

Run the test suite with:

```powershell
python -m pytest branch-analysis-model/tests -q
```
