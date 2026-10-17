# Within-Branch Sales Forecasting & Anomaly Detection

For each branch, the model gives the **expected number of units sold** for every
product for each of the **next 14 days**. Separately, it flags unusual sales by
comparing actual sales with the expected amount. Every branch is judged against its own history. See `PLAN.md`
for the original design.

## Run

```powershell
py -m pip install -r Merchandise/requirements.txt
py Merchandise/run_pipeline.py          # full run -> Merchandise/outputs/
py Merchandise/eda.py                   # exploration printout + figures in Merchandise/outputs/eda/
py Merchandise/evaluate_forecast.py     # validation experiments (first model, NB check, injection tests)
py Merchandise/compare_models.py        # validation: raw vs linear-tree vs baseline-offset LightGBM
py Merchandise/compare_horizons.py      # validation: 1-14 day forecasts, multi-day model vs baseline
py -m unittest discover -s Merchandise/tests -v
```

Outputs in `Merchandise/outputs/`:

| File | Contents |
|---|---|
| `forecast_next_14_days.csv` | Expected units sold per branch × product × day, Jul 1 – 14, with days ahead |
| `category_forecast_next_14_days.csv` | Expected units per branch × category × day, plus whole-branch totals |
| `forecast_2026-07-01.csv` | Tomorrow only (day 1 of the above), next to the usual amount (last 4 same weekdays) |
| `category_forecast_2026-07-01.csv` | Tomorrow's expected units per branch × category and whole branch |
| `report.html` | One tab per branch: tomorrow's expected units, products expected up/down, next 14 days per day and per product, top alerts, actual vs forecast chart |
| `test_forecasts.csv` | Test period: actual units vs expected units (1 day ahead) |
| `test_accuracy_by_days_ahead.csv` | Test accuracy for forecasts made 1 – 14 days in advance, overall and per branch |
| `alerts_test.csv` | Ranked alerts over the test period with profit impact and a one-line message |
| `summary.json` | Headline metrics |

## Method

1. **Daily panel** (`src/panel.py`): branch × product × day from the cleaned file
   (the 132 removed rows were exact duplicates), zero-filled. Split by whole days:
   train to May 15, validation May 16 – Jun 8, test Jun 9 – 30.
2. **Features** (`src/features.py`): lags 1/7/14/28, last-4-same-weekday mean,
   rolling means, branch traffic level, weekday, trend, previous day's price ratio.
   All use data up to the previous day only. A test checks that no feature sees same-day or future
   sales.
3. **Expected units** (`src/forecast.py`): a baseline (recent 28-day level × branch
   weekday pattern × recent branch growth) is the starting point, and one LightGBM
   model (Poisson objective, `log(baseline)` as offset) across all 135 series learns
   adjustments on top of it. The first version predicted raw units and
   under-forecast busy days by up to 9%, because tree models can't predict above the
   values they were trained on. The offset version can. The variant was chosen on
   validation (`compare_models.py`).
4. **Next 14 days** (`src/horizon.py`): for a day *h* days ahead, every sales-based
   input is taken as of the day the forecast is made (no sales after it), while the
   calendar (weekday, date, trend) belongs to the target day. One model covers
   all 14 horizons, with "days ahead" as an input. Tests confirm that no input sees
   sales or prices from after the forecast day.
5. **Alerts** (`src/anomaly.py`): around each expected value, sales are modelled as
   **negative binomial** (spread fitted per branch on validation). A day is flagged
   when the actual sales would be very unlikely under that spread. Checks run at
   three levels:
   - product-day: spikes, drops, zero-sales days that should not be zero
   - branch × category-day: larger counts, so smaller shifts are visible
   - product 7-day window: sustained drops or rises

   Each alert carries its profit impact (`(actual − expected) × unit margin`), and
   alerts are ranked by severity.

## Results on the held-out test period (Jun 9 – 30)

Product-level error (WAPE, lower is better) and bias:

| Branch | Same weekday last week | Weekday average | First model (raw) | **Final model (offset)** |
|---|---|---|---|---|
| Garden | 66% | 56% | 49% | **49%** |
| Harbour | 57% | 49% | 42% | **41%** |
| Uptown | 91% | 79% | 69% | **69%** |
| All | 66% | 57% | 49% | **49%** |
| Bias (all) | −0.6% | −1.5% | −3.3% | **−0.3%** |

Error on each branch's daily total: Garden 11%, Harbour 7%, Uptown 16%.

Forecasts made further in advance (multi-day model, all branches):

| Days ahead | Product error | Branch daily total error | Bias |
|---|---|---|---|
| 1 | 49.1% | 9.9% | −0.4% |
| 7 | 49.1% | 10.0% | −0.3% |
| 14 | 49.2% | 10.0% | 0.0% |

Accuracy hardly changes up to 14 days, and validation showed the same. A review
checked this for leaks and found it genuine. Sales here are driven by a stable
weekday pattern plus steady growth, and day-to-day noise is independent, so recent
days add little beyond level and trend. Beyond about a month the model has no
knowledge of seasons or holidays, so it is not used for that.

The offset model removed the systematic under-forecast. Tuesday bias went from
−8.8% to −1.1%. Wednesday is still over-forecast by 12% on test. That is only 3
days, and validation showed no Wednesday bias, so it was left alone rather than
tuned on test.

Product-level error is close to the floor. Even a forecast that knew each
product's true mean demand would have about 47–51% WAPE, because a product selling
about 2.5 units/day swings by ±1–2 units by chance. Totals are much more
predictable.

Alerts: 57 over 22 days across 3 branches (about 0.9 per branch per day).

**Test-set note:** the test period has been scored several times: first model,
offset model, multi-day model. Every choice was made on validation, so the later
runs look at the test period again but involve no tuning on it.

### What "product error" means

WAPE adds up how far each product's forecast was from its actual sales (too high
or too low), then divides by total units sold. 49% means that for every 100 units
sold, forecasts were off by 49 units in total, which is about 1.3 units per product
per day. Products sell only 2–3 a day, so one unit off is already 30–50%. Misses
in both directions cancel when adding up, which is why branch totals are only
7–16% off and bias is about 0%.

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

- No stock data, so sales are capped by availability. A day a product ran out
  looks like low demand, so the forecast predicts sales, not true demand.
- Six months of history, so there is no yearly seasonality or holiday effect.
