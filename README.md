# Stories Coffee — Business Intelligence

Models that turn six months of Stories coffee shop transactions
(Jan–Jun 2026, branches Garden, Harbour and Uptown) into insights and
recommendations for management.

## Open the website

From the repository root:

```powershell
python start_website.py
```

Open **http://127.0.0.1:8000/**. Keep the terminal running; Ctrl+C stops it.
If port 8000 is occupied, run `python start_website.py --port 8001`.
Only Python is needed to serve the page—no npm install or model training.

## Shared data

This repository contains one source dataset and one canonical chronological split
shared by all models.

```text
stories_transactions.csv       Original data (do not edit)
data/                           Shared train/validation/test files and manifest
data_utils.py                   Shared splitting and basket helpers
scripts/create_team_workspaces.py
tests/test_data_utils.py
```

The shared `data/` folder contains `train.csv`, `validation.csv`, and `test.csv`.
Transactions are split chronologically 70%/15%/15%, and all rows from one
transaction stay together.

**Website folder: [`interface/`](interface/). Main page: [`interface/index.html`](interface/index.html).**

| File | Purpose |
|---|---|
| `start_website.py` | Starts the local website |
| `interface/index.html` | Page layout and sidebar |
| `interface/style.css`, `typography.css`, `workflow.css` | Appearance |
| `interface/app.js`, `synced.js`, `workflow.js` | Pages, model views and controls |
| `interface/data.js` | Saved model results displayed by the website |
| `interface/build_data.py` | Refreshes displayed results from model output files |

After new model outputs arrive, run `python interface/build_data.py`, then refresh
the browser. This refresh needs pandas and the exported model files locally;
it does not train models. Simply opening the website uses the existing `data.js`.

## Modules

| Folder | What it does |
|---|---|
| [Marketing_Expert](Marketing_Expert/README.md) | Product pairings, category patterns and dine-in habits |
| [Merchandise](Merchandise/README.md) | 14-day sales forecast per branch and product, plus unusual-sales alerts |
| [PricingOptimization](PricingOptimization/README.md) | Suggested price per product at each branch |
| [branch-analysis-model](branch-analysis-model/README.md) | Compares how each product performs across branches |
| [decision-model](decision-model/README_decision_rl.md) | Turns model outputs into proposed actions (in progress) | //decision model was not used as implementation wasnt finalized on time
| [interface](interface/README.md) | Local web page that shows the results |

Each folder's README explains how to run it.

## How the decisions were generated

Outputs from Marketing Expert, Merchandise, Pricing Optimization and Branch
Analysis were provided to an LLM (Codex) for review. The LLM used this evidence
to generate the proposed decisions shown on the website, including supporting
reasons, limitations and suggested pilot steps.

These recommendations are saved in `interface/manual_decisions.js` and retained
as a supplementary LLM review on the Decisions page. They were
generated during an assisted review, not by a live LLM service or the untrained
PPO decision engine in `decision-model/`. They are proposals for the owner to
review; no prices or business operations were changed automatically.
## Fallback decisions

If the primary RL decision pipeline is unavailable or fails validation, use
[`fallback_decisions.json`](fallback_decisions.json). It contains deterministic,
human-approved fallback rules and candidate decisions. Forecast-dependent entries
must be refreshed before use when their evidence period has expired.

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
- Fit preprocessing on training data only.
- Feed only boolean product baskets to FP-Growth.
- Use branch and profit after rule generation for filtering and ranking.

The combination model is documented in
[Marketing_Expert/README.md](Marketing_Expert/README.md). Training and final test
evaluation are separate, explicit commands.
