import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import OFFSET_FEATURES, add_features  # noqa: E402


def synthetic_panel(days=180, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days)
    rows = []
    for branch in ["Garden", "Uptown"]:
        for product, cls in [("Espresso", "made_to_order"), ("Brownie", "fresh")]:
            level = np.linspace(2, 4, days) * np.where(dates.dayofweek == 1, 1.6, 1.0)
            rows.append(pd.DataFrame({
                "Branch": branch, "Product": product, "Category": "X", "shelf_class": cls, "Date": dates,
                "units": rng.poisson(level).astype(float), "price": 3.0, "unit_margin": 2.0,
            }))
    return pd.concat(rows, ignore_index=True)


class NoLeakTest(unittest.TestCase):
    def test_features_ignore_today_and_future(self):
        panel = synthetic_panel()
        # After TRAIN_END: the weekday index and price reference are fitted on training days by design,
        # so this checks what matters for forecasting — no feature for day t sees sales from day t onward.
        cutoff = pd.Timestamp("2026-05-25")
        changed = panel.copy()
        after = changed["Date"] >= cutoff
        changed.loc[after, "units"] *= 10
        changed.loc[after, ["price", "unit_margin"]] *= 2
        a, b = add_features(panel), add_features(changed)
        cols = [c for c in OFFSET_FEATURES if c not in ("Branch", "Product", "Category", "shelf_class")]
        mask = a["Date"] <= cutoff  # features on the cutoff day itself must not see that day's sales
        pd.testing.assert_frame_equal(a.loc[mask, cols].reset_index(drop=True),
                                      b.loc[mask, cols].reset_index(drop=True))

    def test_baseline_tracks_weekday_peak(self):
        f = add_features(synthetic_panel()).dropna(subset=["baseline"])
        tue = f[f["weekday"] == 1]["baseline"].mean()
        other = f[f["weekday"] != 1]["baseline"].mean()
        self.assertGreater(tue / other, 1.3)


if __name__ == "__main__":
    unittest.main()
