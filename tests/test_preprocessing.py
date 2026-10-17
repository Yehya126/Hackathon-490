from __future__ import annotations

import unittest

import pandas as pd

from coffee_recommender.features.baskets import BasketEncoder, ProductProfitEstimator


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "Transaction ID": ["T1", "T1", "T2"],
                "Product": ["Coffee", "Cake", "Coffee"],
                "Quantity": [1, 2, 1],
                "Line Gross Profit (USD)": [2.0, 6.0, 4.0],
            }
        )

    def test_unknown_products_do_not_change_training_vocabulary(self) -> None:
        encoder = BasketEncoder().fit(self.train)
        original = self.train.copy(deep=True)
        validation = pd.DataFrame({"Transaction ID": ["V1", "V1"], "Product": ["Coffee", "Tea"]})
        transformed = encoder.transform(validation)
        self.assertEqual(list(transformed.columns), ["Cake", "Coffee"])
        self.assertTrue(transformed.loc["V1", "Coffee"])
        self.assertFalse(transformed.loc["V1", "Cake"])
        pd.testing.assert_frame_equal(self.train, original)

    def test_profit_statistics_are_fitted_on_training_only(self) -> None:
        estimator = ProductProfitEstimator(statistic="median").fit(self.train)
        validation = pd.DataFrame(
            {"Product": ["Coffee"], "Quantity": [1], "Line Gross Profit (USD)": [1000.0]}
        )
        self.assertEqual(float(estimator.transform(validation["Product"]).iloc[0]), 3.0)

    def test_multiple_encoders_can_use_same_input_without_mutation(self) -> None:
        original = self.train.copy(deep=True)
        first = BasketEncoder().fit_transform(self.train)
        second = BasketEncoder().fit_transform(self.train)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(self.train, original)


if __name__ == "__main__":
    unittest.main()

