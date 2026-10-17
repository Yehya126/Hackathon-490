import unittest
from Marketing_Expert.train_all import grade


class ReviewTierTests(unittest.TestCase):
    def test_evidence_and_effect_both_required(self):
        self.assertEqual(grade(30, .05, 1.1), 'Priority to review')
        self.assertEqual(grade(10, .03, 1.01), 'Tentative')
        self.assertEqual(grade(100, .8, .9), 'Weak evidence')
        self.assertEqual(grade(9, .8, 2), 'Weak evidence')
