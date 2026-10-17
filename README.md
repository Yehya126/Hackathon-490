# Coffee Shop Product Recommender

This repository contains one source dataset and identical chronological splits
for the three team members.

```text
stories_transactions.csv       Original data (do not edit)
data_utils.py                   Shared splitting and basket helpers
scripts/create_team_workspaces.py
tests/test_data_utils.py

mohammad/data/                  Mohammad's train/validation/test files
yehya/data/                     Yehya's train/validation/test files
aya/data/                       Aya's train/validation/test files
```

Each member's `data` folder contains `train.csv`, `validation.csv`, and
`test.csv`. The files are identical across members. Transactions are split
chronologically 70%/15%/15%, and all rows from one transaction stay together.

To regenerate the splits:

```powershell
python scripts/create_team_workspaces.py
```

To run the tests:

```powershell
python -m unittest discover -s tests -v
```

Modeling rules:

- Develop with training and validation data only.
- Keep test data untouched until final evaluation.
- Fit preprocessing on training data only.
- Feed only boolean product baskets to FP-Growth.
- Use branch and profit after rule generation for filtering and ranking.

Mohammad's model implementation and commands are documented in
[mohammad/README.md](mohammad/README.md). Training and final test evaluation are
separate, explicit commands.
