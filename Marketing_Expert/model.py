"""Discover combinations and produce owner-facing original-price tables."""

import math
import pandas as pd
from data_utils import make_baskets


DEFAULTS = dict(support=0.005, count=20, confidence=0.2, lift=1.01, profit=0.0)
COLUMNS = ['rank', 'branch', 'combination', 'original_price_usd', 'unit_cost_usd',
           'gross_profit_usd', 'margin_pct', 'transactions', 'support',
           'rule_direction', 'confidence', 'lift', 'ranking_score']


def validate(lines):
    required = ['Transaction ID', 'Branch', 'Product', 'Unit Price (USD)', 'Unit Cost (USD)']
    if not set(required) <= set(lines.columns):
        raise ValueError('Missing required columns')
    if lines.empty or lines[required].isna().any().any():
        raise ValueError('Data must be nonempty and complete')
    for column in required[-2:]:
        if not lines[column].map(math.isfinite).all() or (lines[column] < 0).any():
            raise ValueError(f'Invalid {column}')
    if lines.groupby('Transaction ID')['Branch'].nunique().gt(1).any():
        raise ValueError('Transaction spans branches')


def price_summary(lines):
    """Historical median original unit prices and costs; no discounts applied."""
    validate(lines)
    return lines.groupby('Product')[['Unit Price (USD)', 'Unit Cost (USD)']].median()


def fit_model(train, min_support=0.005, min_branch_transactions=200, max_length=3):
    from mlxtend.frequent_patterns import fpgrowth
    validate(train)
    if not 0 < min_support <= 1 or min_branch_transactions < 1 or max_length < 2:
        raise ValueError('Invalid mining settings')
    scopes = [('Overall', train)] + list(train.groupby('Branch'))
    model = dict(version=2, mining_support=min_support, scopes={}, skipped_branches=[])
    for branch, lines in scopes:
        baskets = make_baskets(lines)[0]
        if branch != 'Overall' and len(baskets) < min_branch_transactions:
            model['skipped_branches'].append(str(branch))
            continue
        frequent = fpgrowth(baskets, min_support=min_support, use_colnames=True, max_len=max_length)
        supports = dict(zip(frequent['itemsets'], frequent['support']))
        prices = price_summary(lines)
        combinations = []
        for items, support in supports.items():
            if len(items) < 2:
                continue
            rules = []
            for target in sorted(items):
                antecedent = items - {target}
                confidence = float(support / supports[antecedent])
                rules.append(dict(antecedent=sorted(antecedent), target=target,
                                  confidence=confidence,
                                  lift=confidence / float(supports[frozenset([target])])))
            price = float(prices.loc[sorted(items), 'Unit Price (USD)'].sum())
            cost = float(prices.loc[sorted(items), 'Unit Cost (USD)'].sum())
            combinations.append(dict(products=sorted(items), price=price, cost=cost,
                                     support=float(support), count=int(round(support * len(baskets))), rules=rules))
        model['scopes'][str(branch)] = dict(transactions=len(baskets), combinations=combinations)
    return model


def combination_table(model, thresholds=None):
    """All qualifying unique itemsets, ranked separately within each scope."""
    limits = DEFAULTS | (thresholds or model.get('thresholds', {}))
    if limits['support'] < model['mining_support']:
        raise ValueError('Table support cannot be below mining support')
    rows = []
    for branch, scope in model['scopes'].items():
        branch_rows = []
        for combo in scope['combinations']:
            profit = combo['price'] - combo['cost']
            if combo['support'] < limits['support'] or combo['count'] < limits['count']:
                continue
            if profit <= 0 or profit < limits['profit'] or combo['price'] <= 0:
                continue
            rules = [r for r in combo['rules'] if r['confidence'] >= limits['confidence'] and r['lift'] >= limits['lift']]
            if not rules:
                continue
            rule = max(rules, key=lambda r: (r['confidence'], r['lift'], r['target']))
            branch_rows.append(dict(branch=branch, combination=' + '.join(combo['products']),
                original_price_usd=combo['price'], unit_cost_usd=combo['cost'], gross_profit_usd=profit,
                margin_pct=100 * profit / combo['price'], transactions=combo['count'], support=combo['support'],
                rule_direction=' + '.join(rule['antecedent']) + ' -> ' + rule['target'],
                confidence=rule['confidence'], lift=rule['lift'], ranking_score=combo['support'] * profit))
        branch_rows.sort(key=lambda r: (-r['ranking_score'], -r['lift'], r['combination']))
        rows.extend(dict(rank=i, **row) for i, row in enumerate(branch_rows, 1))
    return pd.DataFrame(rows, columns=COLUMNS)


def evaluate_table(model, lines):
    """Check frozen combinations and rule directions in a later time period."""
    validate(lines)
    selected = combination_table(model)
    results = []
    for branch, scope in model['scopes'].items():
        subset = lines if branch == 'Overall' else lines.loc[lines['Branch'] == branch]
        baskets = subset.groupby('Transaction ID')['Product'].agg(set).tolist()
        for combo in scope['combinations']:
            match = selected.loc[(selected['branch'] == branch) & (selected['combination'] == ' + '.join(combo['products']))]
            if match.empty:
                continue
            row = match.iloc[0]
            rule = next(r for r in combo['rules'] if ' + '.join(r['antecedent']) + ' -> ' + r['target'] == row['rule_direction'])
            count = sum(set(combo['products']) <= b for b in baskets)
            antecedents = sum(set(rule['antecedent']) <= b for b in baskets)
            targets = sum(rule['target'] in b for b in baskets)
            confidence = count / antecedents if antecedents else None
            lift = confidence / (targets / len(baskets)) if antecedents and targets else None
            results.append(dict(branch=branch, combination=row['combination'], rule_direction=row['rule_direction'],
                period_transactions=len(baskets), transactions=count,
                support=count / len(baskets) if baskets else None, confidence=confidence, lift=lift,
                training_gross_profit_usd=float(row['gross_profit_usd'])))
    return results
