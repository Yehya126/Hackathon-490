"""Transaction-counted dine-in preferences, with a frozen training ranking."""
import pandas as pd


def summarize(lines):
    required = ['Transaction ID', 'Branch', 'Order Type', 'Product']
    if lines[required].isna().any().any():
        raise ValueError('Missing order type, product, branch, or transaction')
    if not set(lines['Order Type']) <= {'Dine-in', 'Takeaway'}:
        raise ValueError('Unrecognized order types')
    if lines.groupby('Transaction ID')[['Branch', 'Order Type']].nunique().gt(1).any().any():
        raise ValueError('Inconsistent transaction metadata')
    transactions = lines.drop_duplicates('Transaction ID')
    baseline = transactions['Order Type'].eq('Dine-in').mean()
    unique = lines.drop_duplicates(['Transaction ID', 'Product'])
    counts = unique.assign(dine_in=unique['Order Type'].eq('Dine-in')).groupby('Product')['dine_in'].agg(['size', 'sum'])
    return baseline, counts


def dine_in_table(train, validation):
    rows = []
    for branch in ['Overall', *sorted(train['Branch'].unique())]:
        earlier = train if branch == 'Overall' else train.loc[train['Branch'] == branch]
        later = validation if branch == 'Overall' else validation.loc[validation['Branch'] == branch]
        baseline, counts = summarize(earlier)
        later_baseline, later_counts = summarize(later)
        candidates = []
        for product, count in counts.iterrows():
            share = count['sum'] / count['size']
            if count['size'] < 20 or share <= .5 or share <= baseline:
                continue
            n = int(later_counts.loc[product, 'size']) if product in later_counts.index else 0
            dine = int(later_counts.loc[product, 'sum']) if n else 0
            later_share = dine / n if n else None
            status = ('Too few later purchases to judge' if n < 20 else
                      'Dine-in preference held up; worth reviewing' if later_share > .5 and later_share > later_baseline else
                      'Earlier dine-in preference did not hold up')
            candidates.append(dict(branch=branch, product=product, training_orders=int(count['size']),
                training_dine_in=int(count['sum']), training_share=share, training_baseline=baseline,
                validation_orders=n, validation_dine_in=dine, validation_share=later_share,
                validation_baseline=later_baseline, assessment=status))
        candidates.sort(key=lambda r: (-(r['training_share'] - baseline), -r['training_orders'], r['product']))
        rows.extend(dict(rank=i, **r) for i, r in enumerate(candidates, 1))
    return pd.DataFrame(rows)


def readable_dine_in(data):
    rows = []
    for _, row in data.iterrows():
        later = row['validation_share']
        rows.append({'Branch': row['branch'], 'Training rank': row['rank'], 'Product': row['product'],
            'Earlier purchases': f"{row['training_dine_in']} of {row['training_orders']} orders were dine-in ({row['training_share']:.0%})",
            'Usual dine-in share: earlier → later': f"{row['training_baseline']:.0%} → {row['validation_baseline']:.0%}",
            'Later purchases': f"{row['validation_dine_in']} of {row['validation_orders']} were dine-in ({later:.0%})" if pd.notna(later) else 'No later purchases',
            'Compared with usual later orders': f"{(later-row['validation_baseline'])*100:+.1f} percentage points dine-in" if pd.notna(later) else 'Cannot judge',
            'What to make of it': row['assessment']})
    return pd.DataFrame(rows, columns=['Branch', 'Training rank', 'Product', 'Earlier purchases',
        'Usual dine-in share: earlier → later', 'Later purchases', 'Compared with usual later orders', 'What to make of it'])
