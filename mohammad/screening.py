"""Conservative review policy, not a significance test or profit guarantee."""
import math


def combination_decision(row):
    reasons = []
    for period in ('train', 'validation'):
        if row[f'transactions_{period}'] < 20:
            reasons.append(f'fewer than 20 purchases in {period}')
        for metric, minimum in [('lift', 1.2), ('confidence', .05)]:
            value = row[f'{metric}_{period}']
            if value is None or not math.isfinite(value) or value < minimum:
                reasons.append(f'{period} {metric} below {minimum}')
    for metric in ('support', 'confidence'):
        if row[f'{metric}_validation'] < .75 * row[f'{metric}_train']:
            reasons.append(f'{metric} fell by more than 25%')
    if 'gross_profit_usd' in row and row['gross_profit_usd'] <= 0:
        reasons.append('no positive original-price gross profit')
    return 'Excluded: ' + '; '.join(reasons) if reasons else 'Include for owner review'


def dine_in_decision(row):
    reasons = []
    for period in ('training', 'validation'):
        if row[f'{period}_orders'] < 50:
            reasons.append(f'fewer than 50 product orders in {period}')
        share, baseline = row[f'{period}_share'], row[f'{period}_baseline']
        if share is None or not math.isfinite(share) or not math.isfinite(baseline):
            reasons.append(f'insufficient {period} evidence')
        elif share <= .5 or share - baseline < .10 - 1e-12:
            reasons.append(f'{period} dine-in preference below the required 10-point advantage')
    return 'Excluded: ' + '; '.join(reasons) if reasons else 'Include for owner review'
