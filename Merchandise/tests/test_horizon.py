import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import add_features  # noqa: E402
from horizon import HORIZON_FEATURES, horizon_frame  # noqa: E402
from test_features import synthetic_panel  # noqa: E402

NUMERIC = [c for c in HORIZON_FEATURES if c not in ("Branch", "Product", "Category", "shelf_class")]


class HorizonNoLeakTest(unittest.TestCase):
    def test_inputs_only_use_data_up_to_the_origin_day(self):
        panel = synthetic_panel()
        cutoff = pd.Timestamp("2026-06-01")  # after TRAIN_END, so training-fitted parts stay fixed
        changed = panel.copy()
        after = changed["Date"] >= cutoff
        changed.loc[after, "units"] *= 10
        changed.loc[after, ["price", "unit_margin"]] *= 2
        a_df, b_df = add_features(panel), add_features(changed)
        for h in (1, 5, 7, 8, 14):
            a, b = horizon_frame(a_df, h), horizon_frame(b_df, h)
            known = a["origin"] < cutoff  # forecast made before any changed day
            pd.testing.assert_frame_equal(a.loc[known, NUMERIC].reset_index(drop=True),
                                          b.loc[known, NUMERIC].reset_index(drop=True), check_names=False)

    def test_same_weekday_inputs_match_target_weekday(self):
        df = add_features(synthetic_panel())
        for h in (3, 10):
            f = horizon_frame(df, h).dropna(subset=["same_weekday_last"])
            s = df.sort_values(["Branch", "Product", "Date"]).set_index(["Branch", "Product", "Date"])["units"]
            k = 7 * ((h + 6) // 7)
            row = f.iloc[200]
            expected = s.loc[(row["Branch"], row["Product"], row["Date"] - pd.Timedelta(days=k))]
            self.assertEqual(row["same_weekday_last"], expected)
            self.assertLessEqual(k, 14)
            self.assertGreaterEqual(k, h)


if __name__ == "__main__":
    unittest.main()
