# Profit-aware coffee-shop recommender

This repository prepares transaction data for a time-aware market-basket
recommender. Model training is intentionally not included yet.

## Data preparation

The raw CSV contains one product line per transaction row. Preparation:

1. validates required columns and parses the transaction timestamp;
2. splits complete transactions chronologically into 70% training, 15%
   validation, and 15% test partitions;
3. fits a product vocabulary and typical unit-profit estimates on training only;
4. transforms each partition into a boolean transaction-product basket matrix.

Run from the repository root:

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_data.py
python scripts/create_team_workspaces.py
python -m unittest discover -s tests -v
```

`prepare_data.py` keeps splits in memory. All models should reuse those same
transaction memberships. The test partition must remain untouched until final
evaluation.

For team development, `create_team_workspaces.py` writes byte-identical split
files under `mohammad/data`, `yehya/data`, and `aya/data`. Each workspace has a
checksum manifest. These files retain the original fields and a `Source Row ID`
for alignment. Association models must use only product-presence baskets; IDs,
dates, branches, quantities, and financial columns are not FP-Growth inputs.

## Structure

```text
src/coffee_recommender/
  data/load_data.py       # schema validation and timestamp parsing
  data/split_data.py      # chronological transaction-level split
  eda/profile.py          # non-mutating EDA summaries
  features/baskets.py     # training-fitted basket/profit transformers
scripts/prepare_data.py   # executable preparation workflow
scripts/create_team_workspaces.py # identical persisted team splits
tests/                    # leakage, alignment, and transformer tests
mohammad/, yehya/, aya/   # team workspaces with matching data splits
```

FP-Growth will later consume only the boolean basket matrices. Identifiers,
timestamps, branch labels, quantities, and financial values are not model
features. Branch is retained only to select records for branch-specific models;
profit is joined after rule generation for recommendation ranking.
