import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from batches import afternoon_topup, two_batch  # noqa: E402


class TwoBatchTest(unittest.TestCase):
    def test_strong_morning_raises_topup(self):
        weak = afternoon_topup(6, 0.4, 0.05, morning_sold=0, on_shelf=0, critical=0.6)
        strong = afternoon_topup(6, 0.4, 0.05, morning_sold=6, on_shelf=0, critical=0.6)
        self.assertGreater(strong, weak)

    def test_leftover_reduces_topup(self):
        empty = afternoon_topup(6, 0.4, 0.05, morning_sold=3, on_shelf=0, critical=0.6)
        stocked = afternoon_topup(6, 0.4, 0.05, morning_sold=3, on_shelf=3, critical=0.6)
        self.assertEqual(stocked, max(empty - 3, 0))

    def test_accounting_is_consistent(self):
        rng = np.random.default_rng(0)
        n = 500
        units = rng.poisson(3, n).astype(float)
        df = pd.DataFrame({
            "Branch": "Garden", "shelf_class": pd.Categorical(["fresh"] * n), "forecast": 3.0,
            "am_share": 0.4, "units": units, "units_am": np.floor(units * 0.4),
            "unit_cost": 1.5, "unit_margin": 2.5,
        })
        s = two_batch(df, "forecast", pd.Series(0.05, index=df.index), 0.9)
        self.assertTrue((s["sold"] <= df["units"]).all())
        self.assertTrue((s["sold"] <= s["prepared"]).all())
        self.assertTrue((s["leftover"] >= 0).all() and (s["missed"] >= 0).all())
        np.testing.assert_allclose(s["prepared"], s["sold"] + s["leftover"])
        np.testing.assert_allclose(s["profit"], s["sold"] * 2.5 - s["leftover"] * 1.5)


if __name__ == "__main__":
    unittest.main()
