"""Run validation, chronological splitting, and training-only preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coffee_recommender.data.load_data import load_transactions
from coffee_recommender.data.split_data import chronological_transaction_split
from coffee_recommender.eda.profile import detailed_training_profile, profile_transactions
from coffee_recommender.features.baskets import BasketEncoder, ProductProfitEstimator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "stories_transactions.csv")
    args = parser.parse_args()

    transactions = load_transactions(args.data)
    splits = chronological_transaction_split(transactions)

    encoder = BasketEncoder()
    train_baskets = encoder.fit_transform(splits.train)
    validation_baskets = encoder.transform(splits.validation)
    test_baskets = encoder.transform(splits.test)
    profit = ProductProfitEstimator().fit(splits.train)

    output = {
        "full_data_screening": profile_transactions(transactions),
        "training_eda": profile_transactions(splits.train),
        "detailed_training_eda": detailed_training_profile(splits.train),
        "split_transactions": {
            "train": len(splits.transaction_ids()["train"]),
            "validation": len(splits.transaction_ids()["validation"]),
            "test": len(splits.transaction_ids()["test"]),
        },
        "basket_shapes": {
            "train": train_baskets.shape,
            "validation": validation_baskets.shape,
            "test": test_baskets.shape,
        },
        "training_profit_products": int(len(profit.product_profit_)),
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()

