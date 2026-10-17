"""Refresh Stories Studio from existing model exports. Never trains a model."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
source = ROOT / 'Marketing_Expert/artifacts/full_data'
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
result['forecasts'] = pd.read_csv(ROOT / 'Merchandise/outputs/forecast_next_14_days.csv').to_dict('records')
branch_csv = ROOT / 'branch-analysis-model/outputs/predictor_output.csv'
result['branch_predictions'] = json.loads(pd.read_csv(branch_csv).to_json(orient='records'))
result['pricing'] = json.loads(pd.read_csv(ROOT / 'PricingOptimization/outputs/price_recommendations.csv').to_json(orient='records'))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


registry_path = ROOT / 'decision-model/config/decisions.json'
registry = read_json(registry_path) or []
definitions = {item['id']: item for item in registry}
evaluation_path = ROOT / 'decision-model/outputs/frozen_test_report/decision_policy_evaluation.json'
synthetic_path = ROOT / 'decision-model/outputs/synthetic_demo/synthetic_policy_report.json'
evaluation = read_json(evaluation_path)
synthetic = read_json(synthetic_path)


def evidence_source(action):
    source = action.get('evidence', {}).get('evidence_source', '')
    if source == 'split_local_branch_variance':
        return 'Branch analysis model'
    if 'combination' in source or action.get('type') in {'bundle_create', 'cross_sell'}:
        return 'Marketing Expert'
    return 'Decision model evidence'


def action_copy(action):
    kind = action.get('type', 'unknown')
    target = action.get('target') or {}
    branch = target.get('branch') or 'Overall'
    products = ([target['product']] if target.get('product') else list(target.get('products') or target.get('bundle_products') or target.get('bundle_product_ids') or []))
    definition = definitions.get(kind, {})
    evidence = action.get('evidence') or {}
    observation = evidence.get('observation') or {}
    expectation = evidence.get('model_expectation') or {}
    residuals = evidence.get('residual_signals') or {}
    product_text = products[0] if len(products) == 1 else ' + '.join(products) if products else 'selected products'
    explanation = 'Selected from the model evidence for management review.'
    next_step = 'Review the source evidence before taking any action.'
    if kind == 'branch_cost_variance_review':
        residual = residuals.get('cost_residual_pct')
        explanation = (f"Recorded unit cost is {abs(residual):.1f}% {'above' if residual >= 0 else 'below'} the cross-branch expectation."
                       if residual is not None else 'The branch-product unit cost differs from the model expectation.')
        next_step = 'Check the recorded cost and operating context before considering a change.'
        title = f'Review {product_text} cost'
    elif kind == 'branch_price_variance_review':
        residual = residuals.get('price_residual_pct')
        explanation = (f"Listed price is {abs(residual):.1f}% {'above' if residual >= 0 else 'below'} the cross-branch expectation."
                       if residual is not None else 'The branch-product listed price differs from the model expectation.')
        next_step = 'Review within-store price performance before any price change.'
        title = f'Review {product_text} price'
    elif kind == 'bundle_create':
        title = f'Review a {product_text} bundle'
        explanation = 'These products appeared together in the supplied association evidence.'
        next_step = 'Owner review first; use a controlled pilot before any rollout.'
    elif kind == 'cross_sell':
        title = f'Review cross-selling {product_text}'
        explanation = 'The supplied association evidence identified this product relationship.'
        next_step = 'Owner review first; measure acceptance in a controlled pilot.'
    else:
        title = definition.get('label') or kind.replace('_', ' ').title()
    return {
        'id': f"{kind}:{branch}:{'|'.join(products) or 'all'}",
        'type': kind,
        'branch': branch,
        'products': products,
        'title': title,
        'explanation': explanation,
        'expected_impact': None,
        'period': None,
        'supporting_evidence': [{
            'source': evidence_source(action),
            'summary': explanation,
        }],
        'assumptions': list(definition.get('guardrails') or []),
        'next_step': next_step,
        'result_kind': 'synthetic_review_proposal',
        'parameters': action.get('parameters') or {},
        'implementation_cost_usd': action.get('implementation_cost_usd'),
        'raw_evidence': {
            'observation': observation,
            'model_expectation': expectation,
            'residual_signals': residuals,
            'prediction_quality': evidence.get('prediction_quality') or {},
            'source': evidence.get('evidence_source'),
        },
    }


synthetic_final = (synthetic or {}).get('final_test_synthetic') or {}
proposals = [action_copy(item) for item in synthetic_final.get('actions') or []]
real_final = (evaluation or {}).get('final_test') or {}
result['decision_output'] = {
    'available': evaluation is not None or synthetic is not None,
    'real_evaluation': {
        'status': (evaluation or {}).get('status'),
        'recommended': real_final.get('recommended'),
        'actions': real_final.get('actions') or [],
        'actual_revenue_usd': real_final.get('actual_business_as_usual_revenue_usd'),
        'actual_gross_profit_usd': real_final.get('actual_business_as_usual_gross_profit_usd'),
        'interpretation': real_final.get('interpretation'),
    },
    'synthetic_demo': {
        'status': (synthetic or {}).get('status'),
        'actions': proposals,
        'action_count': len(proposals),
        'simulated_revenue_change_usd': synthetic_final.get('synthetic_revenue_change_usd'),
        'simulated_gross_profit_change_usd': synthetic_final.get('synthetic_gross_profit_change_usd'),
        'simulated_net_incremental_profit_usd': synthetic_final.get('synthetic_net_incremental_profit_usd'),
        'simulator_support': synthetic_final.get('simulator_support'),
        'assumptions': (synthetic or {}).get('assumptions') or {},
        'limitations': (synthetic or {}).get('limitations') or [],
    },
    'artifact_sources': {
        'real_evaluation': str(evaluation_path.relative_to(ROOT)).replace('\\', '/'),
        'synthetic_demo': str(synthetic_path.relative_to(ROOT)).replace('\\', '/'),
        'registry': str(registry_path.relative_to(ROOT)).replace('\\', '/'),
    },
}
(OUTPUT / 'data.js').write_text('window.STORIES_DATA = ' + json.dumps(result, allow_nan=False) + ';\n', encoding='utf-8')
print('Updated interface/data.js from existing model outputs')
