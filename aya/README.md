# Within-Branch Demand Forecasting & Anomaly Detection

Next-day demand forecasts, prep quantities and unusual-sales alerts for each
branch, with every branch judged against its own history. See `PLAN.md` for the
design.

## Run

```powershell
py -m pip install -r aya/requirements.txt
py aya/run_pipeline.py          # full run -> aya/outputs/
py aya/eda.py                   # exploration printout + figures in aya/outputs/eda/
py aya/evaluate_forecast.py     # validation experiments (models, NB check, injection tests)
py -m unittest discover -s aya/tests -v
```

Outputs in `aya/outputs/`:

| File | Contents |
|---|---|
| `report.html` | One tab per branch: tomorrow's prep list, products expected up/down, top alerts, actual vs forecast chart |
| `prep_plan_2026-07-01.csv` | Forecast, P10–P90 range and recommended prep quantity per branch × product for the next day |
| `alerts_test.csv` | Ranked alerts over the test period with profit impact and a one-line message |
| `test_forecasts.csv` | Test-period actuals, forecasts and ranges |
| `summary.json` | Headline metrics |

## Method

1. **Daily panel** (`src/panel.py`): branch × product × day from the cleaned file
   (the 132 removed rows were exact duplicates), zero-filled. Split by whole days:
   train to May 15, validation May 16 – Jun 8, test Jun 9 – 30.
2. **Features** (`src/features.py`): lags 1/7/14/28, last-4-same-weekday mean,
   rolling means, branch traffic level, weekday, trend, price ratio. All use data up
   to the previous day only.
3. **Forecast** (`src/forecast.py`): one LightGBM model (Poisson objective) across
   all 135 series gives the expected units. Demand is then modelled as **negative
   binomial** around that mean, with dispersion fitted per branch on validation
   residuals. Sales are mildly overdispersed (variance ≈ 1.26 × mean), and NB fits
   better than Poisson on validation.
4. **Prep quantity** (`src/inventory.py`): newsvendor rule, i.e. the NB quantile at
   `margin / (margin + overage cost)`. Overage cost depends on the shelf-life class
   in `src/product_classes.csv`: fresh = full unit cost, made to order = 10%
   (ingredients only), packaged = 5% (holding).
5. **Alerts** (`src/anomaly.py`): tail probability of the actual sales under each
   series' own NB forecast, at three levels:
   - product-day: spikes, drops, zero-sales days that should not be zero
   - branch × category-day: larger counts, so smaller shifts are visible
   - product 7-day window: sustained drops or rises

   Each alert carries its profit impact (`(actual − expected) × unit margin`), and
   alerts are ranked by severity.

## Results on the held-out test period (Jun 9 – 30)

Forecast error (WAPE, lower is better):

| Branch | Same weekday last week | Weekday average | **LightGBM + NB** |
|---|---|---|---|
| Garden | 66% | 56% | **49%** |
| Harbour | 57% | 49% | **42%** |
| Uptown | 91% | 79% | **69%** |
| All | 66% | 57% | **49%** |

The P10–P90 range covers 87–93% of days. With integer counts this is
conservative.

Prep simulation (treating actual sales as demand, all branches):

| Policy | Profit | Waste cost | Missed-sale profit |
|---|---|---|---|
| Prepare what sold same weekday last week | $14,003 | $4,237 | $9,205 |
| Prepare the forecast mean | $17,374 | $2,906 | $7,165 |
| **Prepare the recommended NB quantile** | **$20,153** | $3,299 | **$3,992** |

That is **+44% simulated gross profit** over the last-week rule in three weeks.

Alerts: 49 over 22 days across 3 branches (about 0.7 per branch per day).

## What the detection tests showed

Synthetic anomalies were planted in the validation period:

- Single-day spikes: 100% detected, with a 0.7% false-alert rate on normal days.
- Single-day drops at product level: rarely detectable. With about 2 units/day, a
  zero day happens roughly 13% of the time by chance, so one bad day is
  indistinguishable from noise.
- Sustained 7-day drops (sales cut to 30%): **95% detected**, including 92% of the
  lowest-volume series. This is why the 7-day window and category-level checks
  exist.

## Limitations

- No stock data, so sales are capped by availability. Stockout days look like low
  demand, and the simulation's "demand" is really sales.
- Six months of history, so there is no yearly seasonality or holiday effect.
- Shelf-life classes and overage shares are assumptions; edit
  `src/product_classes.csv` and `OVERAGE_SHARE` in `src/inventory.py`.
