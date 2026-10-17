import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from panel import build_daily_panel, split_by_date  # noqa: E402


def lines(rows):
    columns = ["Date", "Time", "Branch", "Product", "Category", "Quantity", "Unit Price (USD)",
               "Discount Rate", "Unit Cost (USD)", "Line Revenue (USD)", "Line Gross Profit (USD)"]
    data = pd.DataFrame(rows, columns=columns)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Hour"] = pd.to_timedelta(data["Time"]).dt.components.hours
    return data


class PanelTest(unittest.TestCase):
    def setUp(self):
        self.data = lines([
            ("2026-01-01", "0 days 09:00:00", "Garden", "Espresso", "Hot Coffee", 2, 3.0, 0.0, 1.0, 6.0, 4.0),
            ("2026-01-01", "0 days 15:00:00", "Garden", "Espresso", "Hot Coffee", 1, 3.0, 0.0, 1.0, 3.0, 2.0),
            ("2026-01-03", "0 days 10:00:00", "Uptown", "Brownie", "Desserts", 1, 4.0, 0.0, 2.0, 4.0, 2.0),
        ])
        self.panel = build_daily_panel(self.data)

    def test_full_grid_with_zero_fill(self):
        # 2 branches x 2 products x 3 days
        self.assertEqual(len(self.panel), 12)
        self.assertEqual(self.panel["units"].sum(), 4)
        garden = self.panel[(self.panel.Branch == "Garden") & (self.panel.Product == "Espresso")]
        self.assertEqual(garden["units"].tolist(), [3, 0, 0])
        self.assertEqual(garden["last_sale_hour"].iloc[0], 15)

    def test_price_carried_to_zero_days(self):
        garden = self.panel[(self.panel.Branch == "Garden") & (self.panel.Product == "Espresso")]
        self.assertEqual(garden["price"].tolist(), [3.0, 3.0, 3.0])
        self.assertEqual(garden["unit_margin"].iloc[-1], 2.0)

    def test_shelf_class_attached(self):
        classes = self.panel.drop_duplicates("Product").set_index("Product")["shelf_class"]
        self.assertEqual(classes["Espresso"], "made_to_order")
        self.assertEqual(classes["Brownie"], "fresh")

    def test_split_has_whole_days_without_overlap(self):
        dates = pd.date_range("2026-05-10", "2026-06-15")
        panel = pd.DataFrame({"Date": dates})
        train, validation, test = split_by_date(panel)
        self.assertEqual(train.Date.max(), pd.Timestamp("2026-05-15"))
        self.assertEqual(validation.Date.min(), pd.Timestamp("2026-05-16"))
        self.assertEqual(validation.Date.max(), pd.Timestamp("2026-06-08"))
        self.assertEqual(test.Date.min(), pd.Timestamp("2026-06-09"))
        self.assertEqual(len(train) + len(validation) + len(test), len(panel))


if __name__ == "__main__":
    unittest.main()
