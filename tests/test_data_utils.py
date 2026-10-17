import unittest

import pandas as pd

from data_utils import make_baskets, split_data, typical_unit_profit


class DataUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for number in range(20):
            for product in (["Coffee", "Cake"] if number % 2 else ["Coffee"]):
                rows.append({
                    "Transaction ID": f"T{number:02}",
                    "Transaction Timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=number),
                    "Product": product,
                    "Quantity": 1,
                    "Line Gross Profit (USD)": 2.0,
                })
        self.data = pd.DataFrame(rows)

    def test_splits_are_disjoint_complete_and_reproducible(self) -> None:
        first = split_data(self.data)
        second = split_data(self.data)
        ids = [set(getattr(first, name)["Transaction ID"]) for name in ("train", "validation", "test")]
        self.assertTrue(ids[0].isdisjoint(ids[1] | ids[2]))
        self.assertTrue(ids[1].isdisjoint(ids[2]))
        self.assertEqual(set.union(*ids), set(self.data["Transaction ID"]))
        for name in ("train", "validation", "test"):
            pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))

    def test_baskets_learn_products_from_training_only(self) -> None:
        train = self.data.iloc[:10]
        validation = pd.DataFrame({"Transaction ID": ["V1"], "Product": ["Unknown"]})
        train_basket, validation_basket = make_baskets(train, validation)
        self.assertEqual(list(train_basket.columns), list(validation_basket.columns))
        self.assertFalse(validation_basket.any(axis=None))

    def test_profit_uses_training_values(self) -> None:
        self.assertEqual(float(typical_unit_profit(self.data)["Coffee"]), 2.0)


if __name__ == "__main__":
    unittest.main()
