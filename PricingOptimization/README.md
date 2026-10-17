# Pricing Optimization

Recommends a new list price for each product at each branch. Unit costs are fixed;
only the price moves.

## Run

```powershell
py PricingOptimization/pricing.py                         # baseline = recent sales history
py PricingOptimization/pricing.py --baseline merchandise  # baseline = Merchandise model forecast
```

Outputs in `PricingOptimization/outputs/`:

| File | Contents |
|---|---|
| `price_recommendations.csv` | Per branch x product: current and new price, % change, raise/cut/keep, expected units and profit change per day under low/mid/high price sensitivity, break-even elasticity |
| `summary.json` | Baseline accuracy, assumptions, action counts, expected profit change per branch, test-period simulation |

## Method

1. **Baseline demand** at today's prices: average daily units over the last 28 days
   (window chosen on validation from 14/28/56 days), or the Merchandise model's forecast.
2. **Price response**: `units = baseline x (new price / old price) ^ elasticity`.
   The data cannot tell us the elasticity. List prices never change during the six months,
   and branch prices differ by a fixed 5% / -4% that is mixed up with branch traffic.
   The Merchandise model even reacts backwards to price for this reason. So elasticities are
   assumptions per category, with low / mid / high scenarios (weights 25/50/25%), see
   `ELASTICITY` in `pricing.py`.
3. **Search**: for every branch x product, try prices within +/-10% in $0.05 steps and keep
   the one with the best expected profit, with one rule: it must not lose money even under
   the most price-sensitive scenario. Revenue accounts for each item's average discount.

Data: the three split files in `data/`, regrouped by whole days (train to May 15,
validation May 16 - Jun 8, test Jun 9 - 30) so no day is cut in half.

## Results

- Baseline error (per-series daily average): 14.2% on validation, 14.9% on test.
- Actions: Garden 34 raises / 11 keep; Harbour 7 raises / 2 cuts / 36 keep;
  Uptown 10 raises / 1 cut / 34 keep. All new prices stay above cost.
- Expected gross profit change, all branches (today ~$1,207/day):
  **+2.9% mid** (+$35.50/day), +4.4% if customers are less sensitive, +1.0% if more.
- Garden gains most (+8.2% mid) because its unit costs are ~45% higher than the other
  branches (49% margin vs ~66%), so losing a few units to a price rise costs less.
- Test check: prices chosen from data up to Jun 8, applied to real Jun 9-30 sales:
  +3.0% mid (+4.5% low, +1.0% high). This checks the baseline out of sample,
  not the elasticities.
- `break_even_elasticity`: how price-sensitive customers would have to be before a
  recommended raise stops paying off (e.g. -2.2 for Garden Almond Croissant).

## Limitations

- Price response comes from assumptions, not the data. A short price test at one branch
  would replace them with measured values.
- Each product is priced on its own: switching between items and effects on the whole
  basket are not modelled.
- Synthetic data, six months, no competitor prices.
