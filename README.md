# Stories Coffee — Business Intelligence

Models that turn six months of Stories coffee shop transactions
(Jan–Jun 2026, branches Garden, Harbour and Uptown) into insights and
recommendations for management.

## Modules

| Folder | What it does |
|---|---|
| [Marketing_Expert](Marketing_Expert/README.md) | Product pairings, category patterns and dine-in habits |
| [Merchandise](Merchandise/README.md) | 14-day sales forecast per branch and product, plus unusual-sales alerts |
| [PricingOptimization](PricingOptimization/README.md) | Suggested price per product at each branch |
| [branch-analysis-model](branch-analysis-model/README.md) | Compares how each product performs across branches |
| [decision-model](decision-model/README_decision_rl.md) | Turns model outputs into proposed actions (in progress) |
| [interface](interface/README.md) | Local web page that shows the results |

Each folder's README explains how to run it.

## Data

- `stories_transactions.csv` — original data. Do not edit.
- `cleaned_stories_transactions.csv` — cleaned copy.
- Each module's `data/` folder holds `train.csv`, `validation.csv` and
  `test.csv`. They are split by date (70% / 15% / 15%) and are the same
  in every folder.

Regenerate the splits and run the shared tests:

```powershell
python scripts/create_team_workspaces.py
python -m unittest discover -s tests -v
```

## Rules

- Build with train and validation data only.
- Keep test data untouched until final evaluation.
