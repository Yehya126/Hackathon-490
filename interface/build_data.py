"""Refresh the UI's real Mohammad results. Never trains a model."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
source = ROOT / 'mohammad/artifacts/full_data'
raw = pd.read_csv(ROOT / 'stories_transactions.csv')
dates = pd.to_datetime(raw['Date'])
months = len(pd.period_range(dates.min().to_period('M'), dates.max().to_period('M'), freq='M'))
denominators = {str(b): int(d.groupby('Transaction ID')['Product'].nunique().ge(2).sum())
                for b, d in [('Overall', raw), *list(raw.groupby('Branch'))]}
result = {'period': f'{dates.min():%b %Y} – {dates.max():%b %Y}', 'months': months,
          'mode': 'Full-data exploration', 'products': [], 'categories': [], 'dine_in': []}
for name in ['products', 'categories', 'dine_in']:
    frame = pd.read_csv(source / f'{name}.csv')
    for row in frame.to_dict('records'):
        row['orders_month'] = row['dine_in_orders' if name == 'dine_in' else 'transactions'] / months
        if name != 'dine_in':
            row['multi_share'] = row['transactions'] / denominators[row['branch']]
        result[name].append(row)
(OUTPUT / 'data.js').write_text('window.STORIES_DATA = ' + json.dumps(result, allow_nan=False) + ';\n', encoding='utf-8')
print('Updated interface/data.js from existing model outputs')
