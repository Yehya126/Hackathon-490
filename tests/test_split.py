from __future__ import annotations

import unittest

import pandas as pd

from coffee_recommender.data.split_data import chronological_transaction_split


def sample_lines() -> pd.DataFrame:
    rows = []
    for number in range(20):
        for product in ("Coffee", "Cake") if number % 2 else ("Coffee",):
            rows.append(
                {
                    "source_row": len(rows),
                    "Transaction ID": f"T{number:02}",
                    "Transaction Timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=number),
                    "Product": product,
                }
            )
    return pd.DataFrame(rows).set_index("source_row", drop=False)


class SplitTests(unittest.TestCase):
    def test_partitions_are_disjoint_complete_and_aligned(self) -> None:
        original = sample_lines()
        split = chronological_transaction_split(original)
        ids = split.transaction_ids()
        self.assertTrue(set(ids["train"]).isdisjoint(ids["validation"]))
        self.assertTrue(set(ids["train"]).isdisjoint(ids["test"]))
        self.assertTrue(set(ids["validation"]).isdisjoint(ids["test"]))
        combined = pd.concat([split.train, split.validation, split.test]).sort_index()
        pd.testing.assert_frame_equal(combined, original)
        self.assertLess(split.train["Transaction Timestamp"].max(), split.validation["Transaction Timestamp"].min())
        self.assertLess(split.validation["Transaction Timestamp"].max(), split.test["Transaction Timestamp"].min())

    def test_split_is_reproducible(self) -> None:
        frame = sample_lines()
        first = chronological_transaction_split(frame)
        second = chronological_transaction_split(frame)
        for name in ("train", "validation", "test"):
            pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))

    def test_same_timestamp_is_not_split_across_boundaries(self) -> None:
        frame = sample_lines()
        frame.loc[frame["Transaction ID"].isin(["T13", "T14"]), "Transaction Timestamp"] = pd.Timestamp("2026-01-14")
        split = chronological_transaction_split(frame)
        locations = {
            transaction_id: name
            for name, part in (("train", split.train), ("validation", split.validation), ("test", split.test))
            for transaction_id in part["Transaction ID"].unique()
        }
        self.assertEqual(locations["T13"], locations["T14"])


if __name__ == "__main__":
    unittest.main()

