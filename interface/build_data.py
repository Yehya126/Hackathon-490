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
fallback_path = ROOT / 'fallback_decisions.json'
evaluation = read_json(evaluation_path)
synthetic = read_json(synthetic_path)
fallback = read_json(fallback_path)


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


def fallback_evidence_sources(item):
    kind = item.get('decision_id')
    sources = []
    if kind in {'cross_sell', 'dine_in_merchandising_review', 'stop'}:
        sources.append('Marketing Expert')
    if 'historical_forecast_units_14d' in (item.get('evidence') or {}):
        sources.append('Merchandise forecast')
    if kind == 'branch_cost_variance_review':
        sources.append('Branch analysis model')
    if kind == 'sales_anomaly_investigation':
        sources.append('Merchandise alerts')
    return sources or ['Fallback decision evidence']


def fallback_explanation(item):
    kind = item.get('decision_id')
    evidence = item.get('evidence') or {}
    if kind == 'cross_sell':
        return (f"Marketing marked this pairing {evidence.get('assessment', 'for review')} from "
                f"{evidence.get('transactions', 'an unspecified number of')} transactions; "
                f"the observed association lift was {evidence['lift']:.2f}."
                if evidence.get('lift') is not None else
                'Marketing evidence marked this product pairing for review.')
    if kind == 'dine_in_merchandising_review':
        advantage = evidence.get('advantage')
        return (f"Historical branch data showed a {advantage * 100:.1f} percentage-point dine-in advantage."
                if advantage is not None else 'Historical branch data marked this product for dine-in review.')
    if kind == 'branch_cost_variance_review':
        return evidence.get('pattern') or 'Branch analysis found a systematic cost variance worth auditing.'
    if kind == 'sales_anomaly_investigation':
        count = evidence.get('historical_repeated_alerts')
        date = evidence.get('last_historical_alert_date')
        if count is not None and date:
            return f'Historical alerts repeated {count} times, most recently on {date}; current confirmation is still required.'
        return 'Historical alerts suggest an investigation only if the signal repeats in current data.'
    if kind == 'stop':
        return (f"Marketing found {evidence.get('priority_combinations', 0)} branch-specific Priority to review combinations; "
                'tentative combinations were not promoted automatically.')
    return 'This candidate met a deterministic fallback rule.'


def fallback_next_step(item):
    status = item.get('status')
    if status == 'pending_fresh_forecast':
        return 'Refresh the Merchandise forecast, then request owner approval before any pilot.'
    if status == 'pending_fresh_forecast_and_owner_review':
        return 'Refresh the Merchandise forecast and complete owner review before any pilot.'
    if status == 'historical_example_requires_fresh_confirmation':
        return 'Confirm the signal in current data before opening an investigation.'
    if status == 'ready_for_owner_review':
        return 'Ask an owner to review the evidence before starting the audit.'
    if item.get('decision_id') == 'stop':
        return 'Take no action; keep tentative combinations available only for exploration.'
    return 'Request human approval before taking any action.'


def fallback_copy(item):
    kind = item.get('decision_id', 'unknown')
    products = list(item.get('products') or [])
    if item.get('product'):
        products.append(item['product'])
    labels = {
        'cross_sell': 'Cross-sell pilot review',
        'dine_in_merchandising_review': 'Dine-in merchandising review',
        'branch_cost_variance_review': 'Branch cost audit',
        'sales_anomaly_investigation': 'Sales anomaly investigation',
        'stop': 'No supported promotion',
    }
    evidence = item.get('evidence') or {}
    explanation = fallback_explanation(item)
    return {
        'id': item.get('fallback_id'),
        'type': kind,
        'branch': item.get('branch') or 'Overall',
        'products': products,
        'title': labels.get(kind, kind.replace('_', ' ').title()),
        'proposed_action': item.get('recommended_action'),
        'explanation': explanation,
        'expected_impact': None,
        'period': None,
        'supporting_evidence': [{
            'source': source_name,
            'summary': explanation,
        } for source_name in fallback_evidence_sources(item)],
        'assumptions': list(item.get('limitations') or []),
        'next_step': fallback_next_step(item),
        'result_kind': 'fallback_candidate',
        'status': item.get('status'),
        'priority': item.get('priority'),
        'scope': item.get('scope'),
        'rule_direction': item.get('rule_direction'),
        'raw_evidence': evidence,
    }


synthetic_final = (synthetic or {}).get('final_test_synthetic') or {}
proposals = [action_copy(item) for item in synthetic_final.get('actions') or []]
fallback_actions = [fallback_copy(item) for item in (fallback or {}).get('fallback_decisions') or []]
real_final = (evaluation or {}).get('final_test') or {}
result['decision_output'] = {
    'available': evaluation is not None or synthetic is not None or fallback is not None,
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
    'fallback': {
        'name': (fallback or {}).get('name'),
        'generated_on': (fallback or {}).get('generated_on'),
        'purpose': (fallback or {}).get('purpose'),
        'actions': fallback_actions,
        'action_count': len(fallback_actions),
        'activation': (fallback or {}).get('activation') or {},
        'freshness_policy': (fallback or {}).get('freshness_policy') or {},
        'explicitly_disallowed': (fallback or {}).get('explicitly_disallowed_fallbacks') or [],
    },
    'artifact_sources': {
        'real_evaluation': str(evaluation_path.relative_to(ROOT)).replace('\\', '/'),
        'synthetic_demo': str(synthetic_path.relative_to(ROOT)).replace('\\', '/'),
        'fallback': str(fallback_path.relative_to(ROOT)).replace('\\', '/'),
        'registry': str(registry_path.relative_to(ROOT)).replace('\\', '/'),
    },
}
(OUTPUT / 'data.js').write_text('window.STORIES_DATA = ' + json.dumps(result, allow_nan=False) + ';\n', encoding='utf-8')
print('Updated interface/data.js from existing model outputs')
