# Marketing Expert → Yehya's decision model

## Where the results are

Use `Marketing_Expert/artifacts/full_data/` from the repository root.

| File | Use |
| --- | --- |
| `products.csv` | Product combinations, original prices, estimated profit and buying evidence |
| `categories.csv` | Category combinations and buying evidence; no single category price |
| `dine_in.csv` | Products that lean toward dine-in within each branch |
| `product_model.json`, `category_model.json` | Full mined combinations/rules and source provenance |
| `report.html` | Human-readable report, not an integration API |

Ignore `artifacts/validation_review/` and the older files directly inside
`artifacts/` for the current integration. They belong to earlier experiments.

These current results were fitted on ALL 33,887 source transactions from
January–June 2026. They are descriptive signals, not independent forecasts or
validated promotion effects. Former validation/test transactions were included.

## Load results without training

```python
from pathlib import Path
import pandas as pd

# Set this to your repository checkout, not the current script's folder.
repo = Path.cwd()  # example assumes execution from the repository root
output = repo / "Marketing_Expert" / "artifacts" / "full_data"
products = pd.read_csv(output / "products.csv")
categories = pd.read_csv(output / "categories.csv")
dine_in = pd.read_csv(output / "dine_in.csv")

branch = "Garden"
candidate_pairs = products.loc[
    products["branch"].eq(branch)
    & products["assessment"].isin(["Priority to review", "Tentative"])
].copy()
candidate_pairs["orders_per_month"] = candidate_pairs["transactions"] / 6

dine_in_candidates = dine_in.loc[
    dine_in["branch"].eq(branch)
    & dine_in["assessment"].isin(["Priority to review", "Tentative"])
].copy()
dine_in_candidates["dine_in_orders_per_month"] = (
    dine_in_candidates["dine_in_orders"] / 6
)
```

Six months is correct for this export only. Recalculate the period if the source
data changes. Check JSON `provenance.source_sha256` to identify the input snapshot.
The current local folder is exactly `Marketing_Expert` (case matters on Linux).
Some scripts still import `mohammad` and write to that old path after the folder
rename. Fix those references before retraining; reading these existing exports
does not require running or importing the model code.

## Product and category fields

| Field | Meaning |
| --- | --- |
| `branch` | Garden, Harbour, Uptown, or Overall (pooled data) |
| `combination` | Display name of the unordered combination |
| `rank` | Rank within branch and table; not comparable across model types |
| `transactions` | Distinct orders containing all members, including orders with extra items |
| `support` | Such orders divided by ALL orders in the scope; fraction 0–1 |
| `rule_direction` | Which antecedent predicts which additional item |
| `confidence` | Fraction of orders with the antecedent that also contain the consequent |
| `lift` | Confidence divided by the consequent's ordinary purchase rate; above 1 means positive association |
| `assessment` | Review heuristic: Priority to review, Tentative, or Weak evidence |
| `ranking_score` | Products: support × gross profit. Categories: support. Not expected incremental profit |

Product-only fields: `original_price_usd`, `unit_cost_usd`, `gross_profit_usd`,
`margin_pct`. Each combination assumes ONE unit of each product. These are sums
of historical median unit prices/costs, without discounts. Margin is already a
percentage (e.g. 60 means 60%); support/confidence are fractions (0.06 means 6%).

For exact product membership use `product_model.json` → `scopes[branch]` →
`combinations[*].products`. Do not rely on splitting the display name at ` + `;
product names could contain that separator. Each combination has its full
directional `rules` list. The CSV displays the selected highest-confidence rule.

The UI's multi-item share is NOT `support`: divide `transactions` by orders
containing at least two DISTINCT products. The current denominators are:
Overall 15,631; Garden 5,202; Harbour 7,523; Uptown 2,906.
Rows overlap, so do not add their counts or profits into a total opportunity.

## Dine-in fields

`transactions` = all orders containing the product; `dine_in_orders` = those
recorded as dine-in. `confidence` = dine-in share of that product's orders.
`baseline` = usual dine-in share across all orders in that scope.
`advantage` = confidence − baseline; 0.05 means FIVE percentage points.
`lift` = confidence / baseline. These meanings differ from combination confidence.
This table supports in-store promotion ideas, not online-channel inference.

## How the decision model should use these signals

1. Select the relevant branch. Use Overall only as an explicitly labeled fallback;
   never treat Overall and branch rows as separate corroborating evidence.
2. Treat review tiers as candidate filters, not final business decisions.
3. Join exact product names and branch to Aya's demand/pricing outputs. Confirm
   availability, forecast date, price assumptions and margins before proposing an offer.
4. Use YY's branch signals as context: an anomaly may change whether or when to
   try a promotion. It does not establish a causal explanation by itself.
5. Produce a proposed action, rationale, input references, limitations and next
   step. Do not equate purchase association with predicted promotion acceptance.

Current combination priority: >=30 purchases, confidence >=5%, lift >=1.10.
Tentative: >=10 purchases and lift >1. Product combos have positive gross profit.
Dine-in priority: >=50 product orders, majority dine-in, advantage >=5 percentage
points. Tentative: >=20 orders, majority dine-in, any positive advantage.
These are hand-chosen review thresholds, not learned probabilities of success.

Gross profit per sale is NOT additional profit caused by advertising. Any revenue
uplift must be a separately labeled assumption or an estimate supported by a
controlled experiment. Category patterns do not justify every product pairing
within those categories. Current exports do not include inventory, ad costs,
price elasticity or experimentally measured lift.

## Send decisions to the interface

The current interface accepts this JSON through **Connect model output**:

```json
{
  "model": "yehya",
  "decisions": [
    {
      "branch": "Garden",
      "title": "Short proposed action",
      "reason": "Evidence and tradeoff, in plain English",
      "sources": ["Marketing_Expert", "aya", "branch-analysis-model"],
      "status": "Proposed",
      "next_step": "Small pilot and the result to measure"
    }
  ]
}
```

This is a schema example, not an actual recommendation. JSON import currently
stays in the importing browser's local storage. It is not an automatic backend
connection. The UI has legacy contributor labels pending the project rename.

Keep source row references in your own decision output (file, branch, product
membership, source hash). The UI can display the short rationale while retaining
that audit trail in the model's JSON.
