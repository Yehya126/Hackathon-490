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
python -m unittest discover -s tests -v
```

No split files are written. All models should call the same splitter once and
reuse the returned transaction partitions. The test partition must remain
untouched until final evaluation.

## Structure

```text
src/coffee_recommender/
  data/load_data.py       # schema validation and timestamp parsing
  data/split_data.py      # chronological transaction-level split
  eda/profile.py          # non-mutating EDA summaries
  features/baskets.py     # training-fitted basket/profit transformers
scripts/prepare_data.py   # executable preparation workflow
tests/                    # leakage, alignment, and transformer tests
```

FP-Growth will later consume only the boolean basket matrices. Identifiers,
timestamps, branch labels, quantities, and financial values are not model
features. Branch is retained only to select records for branch-specific models;
profit is joined after rule generation for recommendation ranking.
