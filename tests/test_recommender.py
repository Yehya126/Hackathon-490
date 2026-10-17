"""Synthetic checks only; no FP-Growth training or real data evaluation."""
import unittest
from unittest.mock import patch
import pandas as pd
from Marketing_Expert.model import combination_table, evaluate_table, price_summary


class CombinationTests(unittest.TestCase):
    def test_category_presence_and_plain_english_output(self):
        from Marketing_Expert.validation_report import category_lines, association_words
        from data_utils import make_baskets
        data = pd.DataFrame({'Transaction ID': ['A', 'A', 'A'],
                             'Product': ['Latte', 'Espresso', 'Cookie'],
                             'Category': ['Coffee', 'Coffee', 'Bakery']})
        original = data.copy(deep=True)
        baskets = make_baskets(category_lines(data))[0]
        self.assertEqual(baskets.shape, (1, 2))
        self.assertTrue(baskets.loc['A'].all())
        pd.testing.assert_frame_equal(data, original)
        self.assertIn('20% more often', association_words(1.2))
        self.assertIn('Not enough', association_words(float('nan')))

    def setUp(self):
        self.combo = dict(products=['Cake', 'Coffee'], price=9., cost=4., support=.3, count=30,
                          rules=[dict(antecedent=['Coffee'], target='Cake', confidence=.6, lift=1.5),
                                 dict(antecedent=['Cake'], target='Coffee', confidence=.5, lift=1.5)])
        self.model = dict(version=2, mining_support=.005,
                          scopes={'Overall': dict(transactions=100, combinations=[self.combo])})

    def test_unique_table_original_price_profit_and_score(self):
        table = combination_table(self.model)
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]['gross_profit_usd'], 5.)
        self.assertEqual(table.iloc[0]['ranking_score'], 1.5)
        self.assertEqual(table.iloc[0]['rule_direction'], 'Coffee -> Cake')

    def test_thresholds_and_empty_schema(self):
        self.assertTrue(combination_table(self.model, {'count': 31}).empty)
        self.assertTrue(combination_table(self.model, {'profit': 6}).empty)
        self.assertTrue(combination_table(self.model, {'lift': 2}).empty)
        self.assertIn('gross_profit_usd', combination_table(self.model, {'count': 31}).columns)

    def test_prices_ignore_discounts_and_quantity(self):
        lines = pd.DataFrame({'Transaction ID': ['A', 'B'], 'Branch': ['Garden'] * 2,
                              'Product': ['Cake'] * 2, 'Unit Price (USD)': [6., 6.],
                              'Unit Cost (USD)': [2., 2.], 'Quantity': [2, 1],
                              'Line Gross Profit (USD)': [1., 1.]})
        original = lines.copy(deep=True)
        summary = price_summary(lines)
        self.assertEqual(summary.loc['Cake', 'Unit Price (USD)'], 6.)
        pd.testing.assert_frame_equal(lines, original)

    def test_later_period_checks_frozen_direction(self):
        lines = pd.DataFrame({'Transaction ID': ['A', 'A', 'B'], 'Branch': ['Garden'] * 3,
                              'Product': ['Coffee', 'Cake', 'Coffee'],
                              'Unit Price (USD)': [100.] * 3, 'Unit Cost (USD)': [1.] * 3})
        row = evaluate_table(self.model, lines)[0]
        self.assertEqual(row['support'], .5)
        self.assertEqual(row['confidence'], .5)
        self.assertEqual(row['lift'], 1.)
        self.assertEqual(row['training_gross_profit_usd'], 5.)

    def test_help_never_trains(self):
        from Marketing_Expert.run import main
        with patch('sys.argv', ['run.py', '--help']), patch('Marketing_Expert.run.fit_model') as fit:
            with self.assertRaises(SystemExit):
                main()
            fit.assert_not_called()
