"""Fit descriptive models on ALL source transactions and publish an owner report."""
import hashlib
import json
from html import escape
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from mohammad.model import fit_model, combination_table
from mohammad.dine_in import summarize
from mohammad.run import save


def grade(count, confidence, lift):
    """Transparent starting heuristics, not a statistical or commercial guarantee."""
    if count >= 30 and confidence >= .05 and lift >= 1.1:
        return 'Priority to review'
    if count >= 10 and lift > 1:
        return 'Tentative'
    return 'Weak evidence'


def combinations(data, category=False):
    model = fit_model(data, min_support=.0005)
    model['training_scope'] = 'Entire source dataset; no independent evaluation'
    model['thresholds'] = dict(support=.0005, count=10, confidence=0., lift=0., profit=0.)
    table = combination_table(model)
    table['assessment'] = [grade(r.transactions, r.confidence, r.lift) for r in table.itertuples()]
    if category:
        table = table.drop(columns=['original_price_usd', 'unit_cost_usd', 'gross_profit_usd', 'margin_pct'])
        table['ranking_score'] = table['support']
    priority = {'Priority to review': 0, 'Tentative': 1, 'Weak evidence': 2}
    table['_tier'] = table['assessment'].map(priority)
    table = table.sort_values(['branch', '_tier', 'ranking_score', 'combination'], ascending=[True, True, False, True]).drop(columns='_tier')
    table['rank'] = table.groupby('branch').cumcount() + 1
    return model, table


def dine_in(data):
    rows = []
    for branch, lines in [('Overall', data), *list(data.groupby('Branch'))]:
        baseline, counts = summarize(lines)
        for product, count in counts.iterrows():
            n, yes = int(count['size']), int(count['sum'])
            share = yes / n
            advantage = share - baseline
            if baseline == 0:
                continue
            tier = ('Priority to review' if n >= 50 and share > .5 and advantage >= .05 else
                    'Tentative' if n >= 20 and share > .5 and advantage > 0 else 'Weak evidence')
            rows.append(dict(branch=branch, product=product, transactions=n, dine_in_orders=yes,
                             confidence=share, baseline=baseline, advantage=advantage,
                             lift=share / baseline, assessment=tier))
    table = pd.DataFrame(rows)
    table['_tier'] = table['assessment'].map({'Priority to review': 0, 'Tentative': 1, 'Weak evidence': 2})
    table = table.sort_values(['branch', '_tier', 'advantage', 'transactions'], ascending=[True, True, False, False]).drop(columns='_tier')
    table['rank'] = table.groupby('branch').cumcount() + 1
    return table


def section(title, data, kind, months, multi_item_orders=None):
    rows = []
    for r in data.to_dict('records'):
        if kind == 'dine':
            description = f"Dine-in preference is {r['advantage']*100:+.1f} percentage points compared with the usual branch mix."
            monthly = r['dine_in_orders'] / months
            cells = [r['rank'], r['product'], f'{monthly:.1f}', f"{r['confidence']:.1%}", description, r['assessment']]
        else:
            description = ((f"{(r['lift']-1)*100:.1f}% more" if r['lift'] >= 1 else f"{(1-r['lift'])*100:.1f}% less")
                           + ' often than expected from individual popularity.')
            money = 'Depends on chosen products' if kind == 'category' else f"${r['original_price_usd']:.2f} price / ${r['gross_profit_usd']:.2f} gross profit"
            monthly = r['transactions'] / months
            denominator = multi_item_orders[r['branch']]
            share = f"{r['transactions'] / denominator:.1%}" if denominator else 'N/A'
            cells = [r['rank'], r['combination'], f'{monthly:.1f}', share, money + '. ' + description, r['assessment']]
        attrs = f'data-branch="{escape(str(r["branch"]), quote=True)}" data-count="{monthly}" data-lift="{r["lift"]}" data-tier="{escape(r["assessment"], quote=True)}"'
        rows.append('<tr ' + attrs + '>' + ''.join('<td>' + escape(str(c)) + '</td>' for c in cells) + '</tr>')
    branches = ''.join(f'<option>{escape(str(b))}</option>' for b in ['Overall', *sorted(set(data.branch)-{'Overall'})])
    return f'''<section><h2>{title}</h2><div class="controls">
    <label>Branch <select class="branch">{branches}</select></label>
    <label>Show <select class="tier"><option value="all">All candidates</option><option value="review">Priority + tentative</option><option value="priority">Priority only</option></select></label>
    <label>Minimum orders/month <input class="count" type="number" min="0" step="0.1" value="0"></label>
    <label>Minimum association ratio <input class="lift" type="number" min="0" step="0.1" value="0"></label></div>
    <p class="status" aria-live="polite"></p><div class="scroll"><table><thead><tr>
    <th>Rank</th><th>Suggestion</th><th>{'Dine-in orders/month' if kind == 'dine' else 'Orders/month'}</th>{'<th title="Dine-in orders containing this product divided by all orders containing this product, within the selected branch">Dine-in share of product orders</th>' if kind == 'dine' else '<th title="Orders containing this combination divided by all orders with at least two distinct products, within the selected branch. Orders with additional products also count.">Share of multi-item orders</th>'}<th>What the data says</th><th>Review tier</th>
    </tr></thead><tbody>{''.join(rows)}</tbody></table></div><button class="more">View 10 more</button> <button class="less">Show top 10</button></section>'''


def main():
    source = ROOT / 'stories_transactions.csv'
    data = pd.read_csv(source)
    output = ROOT / 'mohammad' / 'artifacts' / 'full_data'
    model, products = combinations(data)
    category_data = data.copy()
    if category_data['Category'].isna().any():
        raise ValueError('Missing category labels')
    category_data['Product'] = category_data['Category']
    category_model, categories = combinations(category_data, category=True)
    dining = dine_in(data)
    provenance = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      rows=len(data), transactions=int(data['Transaction ID'].nunique()),
                      evaluation='None. All observations used for fitting, including former validation/test periods.')
    for name, fitted in [('product_model', model), ('category_model', category_model)]:
        fitted['provenance'] = provenance
        save(output / f'{name}.json', fitted)
    for name, table in [('products', products), ('categories', categories), ('dine_in', dining)]:
        table.to_csv(output / f'{name}.csv', index=False)
        print(name, table.groupby('assessment').size().to_dict())
    render_report(data, products, categories, dining, output)


def render_report(data, products, categories, dining, output):
    multi_item_orders = {
        str(branch): int(lines.groupby('Transaction ID')['Product'].nunique().ge(2).sum())
        for branch, lines in [('Overall', data), *list(data.groupby('Branch'))]
    }
    dates = pd.to_datetime(data['Date'])
    months = len(pd.period_range(dates.min().to_period('M'), dates.max().to_period('M'), freq='M'))
    provenance = dict(transactions=int(data['Transaction ID'].nunique()))
    header = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coffee shop suggestions</title>
    <style>body{{font:16px system-ui;color:#183140;margin:30px;background:#f7fafb}}section{{background:white;padding:22px;margin:24px 0;border-radius:12px}}table{{border-collapse:collapse;min-width:1000px;width:100%}}th,td{{padding:12px;border-bottom:1px solid #dde5e9;text-align:left}}th{{background:#eaf0f4}}.scroll{{overflow:auto}}.controls{{display:flex;gap:16px;flex-wrap:wrap}}label{{display:flex;flex-direction:column;gap:5px}}input{{width:130px}}select,input,button{{font:inherit;padding:8px}}button{{margin-top:16px;cursor:pointer}}[hidden]{{display:none!important}}p{{max-width:1100px;line-height:1.6}}</style></head><body>
    <h1>Coffee shop suggestions</h1><p>Trained on all {len(data):,} lines across {provenance['transactions']:,} transactions. These are observed buying patterns, not validated predictions. Former validation and test transactions are included; future independent evaluation needs new transactions.</p>
    <p><b>How we separate suggestions:</b> Product/category <b>priority</b> requires at least 30 purchases together, 5% observed confidence and an association ratio of 1.10 (10% above individual popularity). <b>Tentative</b> requires at least 10 purchases and a ratio above 1. Otherwise evidence is weak. Product suggestions have positive original-price gross profit.</p>
    <p>For dine-in, <b>priority</b> requires at least 50 product orders, a majority dine-in, and a dine-in share 5 percentage points above the branch baseline. <b>Tentative</b> requires at least 20 orders, a majority dine-in and any positive advantage. These are adjustable review heuristics, not significance tests or proven profitability. All tiers are visible by default.</p>
    <p>Orders/month is the historical average over {months} calendar months ({dates.min():%b %Y}–{dates.max():%b %Y}), including months with no matching orders. Dine-in rows count only dine-in purchases. This is not a monthly forecast. Prices/profits use historical median original unit prices/costs, one of each product, without discounts. Within tiers, combinations rank by frequency × profit; categories by frequency; dine-in by advantage over the baseline.</p>'''
    script = '''<script>document.querySelectorAll('section').forEach(s=>{let limit=10;const rows=[...s.querySelectorAll('tbody tr')];function render(){const tier=s.querySelector('.tier').value;const selected=rows.filter(r=>r.dataset.branch===s.querySelector('.branch').value&&Number(r.dataset.count)>=Number(s.querySelector('.count').value)&&Number(r.dataset.confidence)>=Number(s.querySelector('.confidence').value)&&Number(r.dataset.lift)>=Number(s.querySelector('.lift').value)&&(tier==='all'||tier==='review'&&r.dataset.tier!=='Weak evidence'||tier==='priority'&&r.dataset.tier==='Priority to review'));rows.forEach(r=>r.hidden=true);selected.slice(0,limit).forEach(r=>r.hidden=false);s.querySelector('.status').textContent=`Showing ${Math.min(limit,selected.length)} of ${selected.length} matching suggestions.`;s.querySelector('.more').hidden=limit>=selected.length;s.querySelector('.less').hidden=limit<=10;}s.querySelectorAll('input,select').forEach(e=>e.addEventListener('input',()=>{limit=10;render();}));s.querySelector('.more').onclick=()=>{limit+=10;render();};s.querySelector('.less').onclick=()=>{limit=10;render();};render();});</script>'''
    script = script.replace("&&Number(r.dataset.confidence)>=Number(s.querySelector('.confidence').value)", '')
    header += '<p><b>Share of multi-item orders:</b> percentage of orders containing the combination out of all orders with at least two distinct products in the selected branch. Both combination tables use this same denominator, including orders with multiple products from one category. Orders containing extra products also count; rows overlap, so percentages need not add to 100%.</p>'
    (output / 'report.html').write_text(header + section('1. Product combinations', products, 'product', months, multi_item_orders) + section('2. Category patterns', categories, 'category', months, multi_item_orders) + section('3. Dine-in promotion ideas', dining, 'dine', months) + script + '</body></html>', encoding='utf-8')
    print(output / 'report.html')


if __name__ == '__main__':
    main()
