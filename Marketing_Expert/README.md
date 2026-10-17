# Product combination tables

**For Yehya:** see [INTEGRATION.md](INTEGRATION.md) for current output locations,
CSV field meanings, loading examples and how to consume these signals in the
decision model. Outputs are now under `Marketing_Expert/artifacts/full_data/`.
The commands below retain older `mohammad` paths; update rename references before
using them to rerun training.

## Current workflow: full-data descriptive suggestions

Run `python mohammad/train_all.py` from the repository root. This fits on the
entire original dataset and creates `artifacts/full_data/report.html`, three
CSV tables, and product/category models. The page shows ten rows per table,
branch selection, observed confidence, review tiers and adjustable filters.
Repeated runs overwrite only this full-data output directory's named artifacts.

Priority combinations require 30 purchases, 5% confidence and lift of 1.10;
tentative combinations require 10 purchases and lift above 1. Dine-in priority
requires 50 product orders, majority dine-in and a five-percentage-point advantage
over the branch baseline; tentative requires 20 orders and any positive advantage.
These are review heuristics. Lower tiers remain visible by default.

There is no validation for this model. Former validation/test periods are now
training observations; future independent performance checks require new data.
The older workflow and artifacts below are retained as historical experiments.

To generate an exploratory side-by-side validation report:

```powershell
python mohammad/validation_report.py
```

Open `artifacts/validation_review/report.html`. This mines training candidates
at 0.1% support with at least 10 training purchases, including weak associations
for inspection, and compares their frozen directions against validation data.
It preserves the original model and never reads test data. Review labels are
heuristics, not statistical significance or business approval.

The report now contains two plain-English tables: product combinations and
category patterns, each including Overall and branch rows. Category presence
counts once per transaction; repeated products from one category do not inflate
counts. Category rows omit prices/profit because those depend on the specific
products chosen. Category ranking uses training frequency. Detailed numeric
results are saved as `product_comparison.csv` and `category_comparison.csv`.

FP-Growth finds frequent combinations for the owner to review. Nothing trains
on import. No model has been trained as part of this implementation.

From the repository root, when ready:

```powershell
python -m pip install -r mohammad/requirements.txt
python mohammad/run.py train
python mohammad/run.py validate
```

Training writes `artifacts/model.json`, `combinations.csv`, and a readable
`combinations.html`. Each unique combination appears once per branch or Overall
scope, with original-price total, cost, gross profit, margin, purchase count,
support, directional confidence/lift, and rank. No fixed top-N limit is applied.
Branches with fewer than 200 transactions are recorded as skipped; overall
statistics are never mislabeled as branch evidence.

Prices and costs are training medians per product within each scope. Each combo
contains ONE unit of each product. Profit is the sum of original unit prices
minus the sum of unit costs. Discounts and recorded line profit are not used.
Historical medians are estimates, not a verified current menu price. The owner
chooses actual offers and prices.

Rows rank by `training support × original-price gross profit per combo` within
each scope. This balances observed frequency and margin; it is not incremental
profit or a sales forecast. Overlapping combinations must not be summed into a
profit forecast. Confidence/lift refer to the displayed direction; of the rules
meeting thresholds, the one with highest confidence is displayed.

Default review filters: support >= 0.005, purchase count >= 20, confidence >= 0.2,
lift >= 1.01, positive gross profit. These are starting settings, not established
business break-even points. All passing pairs/triples are exported. Adjust using
validation evidence and owner priorities, for example:

```powershell
python mohammad/run.py train --min-count 30 --min-profit 4 --min-support 0.01 --min-confidence 0.3 --min-lift 1.2 --max-length 3
```

Archive an existing artifacts folder before a new training experiment. Shared
data splits remain unchanged. All learning and pricing use training data only.
Validation reports later purchase frequency, confidence, and lift for the frozen
combinations/directions; it does not generate new combinations or update prices.
There is no automatic threshold optimizer: basket-completion accuracy no longer
represents the owner's objective. Review validation stability to choose filters.

After choices are final, explicitly run the untouched test evaluation once:

```powershell
python mohammad/run.py evaluate-test
```

Reports contain one entry per selected combination/scope. Undefined metrics are
null when a scope or antecedent is absent. Test evidence must not be used to tune
filters. `python mohammad/run.py export` regenerates tables from the saved model.

Duplicate lines remain preserved: basket presence collapses them, but they can
affect price medians. Confirm duplicate semantics before training. Financial
values, branch, and identifiers are never FP-Growth input features.
