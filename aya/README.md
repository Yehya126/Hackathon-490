# Within-Branch Demand Forecasting & Anomaly Detection

Next-day demand forecasts, prep quantities and unusual-sales alerts for each
branch, with every branch judged against its own history. See `PLAN.md` for the
design.

## Run

```powershell
py -m pip install -r aya/requirements.txt
py aya/run_pipeline.py          # full run -> aya/outputs/
py aya/eda.py                   # exploration printout + figures in aya/outputs/eda/
py aya/evaluate_forecast.py     # validation experiments (first model, NB check, injection tests)
py aya/compare_models.py        # validation: raw vs linear-tree vs baseline-offset LightGBM
py aya/choose_batches.py        # validation: two-batch cutoff hour and morning batch size
py -m unittest discover -s aya/tests -v
```

Outputs in `aya/outputs/`:

| File | Contents |
|---|---|
| `report.html` | One tab per branch: tomorrow's prep list, products expected up/down, top alerts, actual vs forecast chart |
| `prep_plan_2026-07-01.csv` | Forecast, P10–P90 range, one-batch prep quantity and fresh morning batch per branch × product for the next day |
| `category_plan_2026-07-01.csv` | Expected units and range per branch × category and whole branch (staffing / ingredients) |
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
3. **Forecast** (`src/forecast.py`): a baseline (recent 28-day level × branch weekday
   pattern × recent branch growth) is the starting point, and one LightGBM model
   (Poisson objective, `log(baseline)` as offset) across all 135 series learns
   adjustments on top of it. The first version predicted raw units and
   under-forecast busy days by up to 9%, because tree models can't predict above the
   values they were trained on. The offset version can. Demand is then modelled as **negative
   binomial** around that mean, with dispersion fitted per branch on validation
   residuals. Sales are mildly overdispersed (variance ≈ 1.26 × mean), and NB fits
   better than Poisson on validation.
4. **Prep quantity** (`src/inventory.py`): newsvendor rule, i.e. the NB quantile at
   `margin / (margin + overage cost)`. Overage cost depends on the shelf-life class
   in `src/product_classes.csv`: fresh = full unit cost, made to order = 10%
   (ingredients only), packaged = 5% (holding).
5. **Two batches for fresh items** (`src/batches.py`): a morning batch at opening, then
   a 15:00 top-up. Morning sales update the afternoon expectation through the same
   negative binomial model, and the top-up is sized to the remaining demand. The
   cutoff and batch size were chosen on validation (`choose_batches.py`).
6. **Category plan**: tomorrow's expected units per branch × category and branch
   total. The error is far lower at these levels (validation: 24% per category, 10%
   per branch vs 51% per product), so staffing and ingredient orders should be
   planned there.
7. **Alerts** (`src/anomaly.py`): tail probability of the actual sales under each
   series' own NB forecast, at three levels:
   - product-day: spikes, drops, zero-sales days that should not be zero
   - branch × category-day: larger counts, so smaller shifts are visible
   - product 7-day window: sustained drops or rises

   Each alert carries its profit impact (`(actual − expected) × unit margin`), and
   alerts are ranked by severity.

## Results on the held-out test period (Jun 9 – 30)

Forecast error (WAPE, lower is better) and bias:

| Branch | Same weekday last week | Weekday average | First model (raw) | **Final model (offset)** |
|---|---|---|---|---|
| Garden | 66% | 56% | 49% | **49%** |
| Harbour | 57% | 49% | 42% | **41%** |
| Uptown | 91% | 79% | 69% | **69%** |
| All | 66% | 57% | 49% | **49%** |
| Bias (all) | −0.6% | −1.5% | −3.3% | **−0.3%** |

The offset model removed the systematic under-forecast. Tuesday bias went from
−8.8% to −1.1%. Wednesday is still over-forecast by 12% on test. That is only 3
days, and validation showed no Wednesday bias, so it was left alone rather than
tuned on test.

Product-level error is close to the floor. Even a forecast that knew each
product's true mean demand would have about 47–51% WAPE, because a product selling
about 2.5 units/day swings by ±1–2 units by chance.

The P10–P90 range covers 89–93% of days.

Prep simulation (treating actual sales as demand, all branches):

| Policy | Profit | Waste cost | Missed-sale profit |
|---|---|---|---|
| Prepare what sold same weekday last week | $14,003 | $4,237 | $9,205 |
| Prepare the forecast mean | $17,640 | $3,013 | $6,791 |
| Prepare the recommended NB quantile (one batch) | $20,278 | $3,362 | $3,804 |
| **Recommended + fresh items in two batches** | **$20,534** | $3,877 | **$3,033** |

That is **+47% simulated gross profit** over the last-week rule. The second batch
adds a little more ($256) by turning missed sales into sales, at slightly higher
waste.

Alerts: 57 over 22 days across 3 branches (about 0.9 per branch per day).

**Test-set note:** the first (raw) model was scored on this test period before the
offset change. The offset model, two-batch settings and everything else were
chosen on validation only, so this run is the second look at the test period but
involved no tuning on it.

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
