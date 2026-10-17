"""Explore training candidates against validation, without reading test data."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from Marketing_Expert.model import combination_table, evaluate_table, fit_model
from Marketing_Expert.run import read_split, save
from Marketing_Expert.dine_in import dine_in_table, readable_dine_in
from Marketing_Expert.screening import combination_decision, dine_in_decision


def verdict(row):
    reasons = []
    if pd.isna(row['lift_validation']) or pd.isna(row['confidence_validation']):
        return 'Insufficient validation evidence: rule metrics undefined'
    if row['lift_train'] <= 1 or row['lift_validation'] <= 1:
        reasons.append('No positive association in both periods')
    if row['transactions_validation'] < 10:
        reasons.append('Fewer than 10 validation purchases')
    if row['support_validation'] < row['support_train'] * .5:
        reasons.append('Frequency fell by more than half')
    if row['confidence_validation'] < row['confidence_train'] * .5:
        reasons.append('Confidence fell by more than half')
    return '; '.join(reasons) if reasons else 'Promising for review (not proven)'


def main():
    train, validation = read_split('train'), read_split('validation')
    if set(train['Transaction ID']) & set(validation['Transaction ID']):
        raise ValueError('Overlapping transactions')
    timestamp = lambda d: pd.to_datetime(d['Date'] + ' ' + d['Time'])
    if timestamp(train).max() >= timestamp(validation).min():
        raise ValueError('Periods are not chronological')
    output = Path(__file__).resolve().parent / 'artifacts' / 'validation_review'
    products = compare(train, validation, output, 'product')
    categories = compare(category_lines(train), category_lines(validation), output, 'category')
    dine_in = dine_in_table(train, validation)
    dine_in.to_csv(output / 'dine_in_comparison.csv', index=False)
    write_report(products, categories, output, dine_in)
    print(f'Product rows: {len(products)}; category rows: {len(categories)}')
    print('Category assessments:', categories['assessment'].value_counts().to_dict())
    print(f'Report: {output / "report.html"}')


def category_lines(lines):
    """Use category presence; repeated categories within an order count once."""
    if 'Category' not in lines or lines['Category'].isna().any():
        raise ValueError('Complete category labels are required')
    result = lines.copy()
    result['Product'] = result['Category']
    return result


def compare(train, validation, output, level):
    model = fit_model(train, min_support=.001)
    # Diagnostic candidates, including weak associations, not approved offers.
    model['thresholds'] = dict(support=.001, count=10, confidence=0., lift=0., profit=0.)
    table = combination_table(model)
    later = pd.DataFrame(evaluate_table(model, validation))
    if table.empty:
        raise ValueError(f'No {level} candidates at exploratory thresholds')
    combined = table.merge(later, on=['branch', 'combination', 'rule_direction'],
                           suffixes=('_train', '_validation'), validate='one_to_one')
    combined['assessment'] = combined.apply(verdict, axis=1)
    if level == 'category':
        # Category prices vary by product: do not present a fabricated combo price.
        combined = combined.drop(columns=['original_price_usd', 'unit_cost_usd', 'gross_profit_usd',
                                           'margin_pct', 'training_gross_profit_usd', 'ranking_score'])
        combined = combined.sort_values(['branch', 'support_train', 'combination'], ascending=[True, False, True])
        combined['rank'] = combined.groupby('branch').cumcount() + 1
    save(output / f'{level}_candidate_model.json', model)
    combined.to_csv(output / f'{level}_comparison.csv', index=False)
    return combined


def association_words(value):
    if pd.isna(value):
        return 'Not enough purchases to judge'
    if value > 1:
        return f'{(value - 1) * 100:.0f}% more often than expected from individual popularity'
    return f'{(1 - value) * 100:.0f}% less often than expected from individual popularity'


def readable_table(data, category=False):
    rows = []
    for _, row in data.iterrows():
        record = {'Branch': row['branch'], 'Training rank': row['rank'],
                  'Categories bought together' if category else 'Products bought together': row['combination']}
        if not category:
            record['Original price / estimated gross profit'] = f"${row['original_price_usd']:.2f} / ${row['gross_profit_usd']:.2f}"
        for suffix, label in [('train', 'Earlier orders'), ('validation', 'Later orders (validation)')]:
            record[label] = f"{row['transactions_' + suffix]:.0f} purchases together; {row['support_' + suffix] * 1000:.1f} in every 1,000 orders"
        record['Buying relationship: earlier → later'] = association_words(row['lift_train']) + ' → ' + association_words(row['lift_validation'])
        direction = row['rule_direction'].split(' -> ')
        confidence = row['confidence_validation']
        record['In plain English (later orders)'] = (f"Among orders with {direction[0]}, about {confidence * 100:.1f} in 100 also included {direction[1]}."
                                                     if pd.notna(confidence) else 'Too few relevant orders to judge.')
        record['What to make of it'] = row['assessment'].replace('No positive association in both periods', 'The pairing did not beat individual popularity in both periods').replace('Promising for review (not proven)', 'Passed basic review checks; still exploratory')
        rows.append(record)
    return pd.DataFrame(rows)


def write_report(products, categories, output, dine_in=None):
    products, categories = products.copy(), categories.copy()
    for name, data in [('product', products), ('category', categories)]:
        data['assessment'] = data.apply(combination_decision, axis=1)
        data.to_csv(output / f'{name}_screened.csv', index=False)
    if dine_in is not None:
        dine_in = dine_in.copy()
        dine_in['assessment'] = dine_in.apply(dine_in_decision, axis=1)
        dine_in.to_csv(output / 'dine_in_screened.csv', index=False)
    def table_section(data, category=False, ready=False):
        view = data if ready else readable_table(data, category=category)
        return ('<section class="results"><label>Branch: <select><option>Overall</option>'
                '<option>Garden</option><option>Harbour</option><option>Uptown</option></select></label>'
                '<label> <input type="checkbox" class="rejected"> Show rejected candidates</label>'
                '<p class="count" aria-live="polite"></p><div class="table-scroll">'
                + view.to_html(index=False, escape=True) + '</div>'
                '<button type="button" class="more">View more</button> '
                '<button type="button" class="less">Show top 10</button></section>')
    sections = ['<h2>1. Product combinations</h2><p>Ranked within each branch by earlier purchase frequency and estimated gross profit.</p>'
                + table_section(products),
                '<h2>2. Category patterns</h2><p>Ranked within each branch by earlier purchase frequency. A category can contain many products; '
                'there is no single price or profit for a category pairing. These patterns do not prove that every product pairing within them works.</p>'
                + table_section(categories, category=True)]
    if dine_in is not None:
        sections.append('<h2>3. Products leaning toward dine-in</h2><p>Products selected using earlier orders: '
                        'at least 20 purchases, more than half dine-in, and a dine-in share above the branch baseline. '
                        'Ranked by how far that share exceeds the earlier baseline. Later orders check the same products; '
                        '20 later purchases are required for a provisional assessment. Counts are orders, not quantities. '
                        'These are recorded dine-in purchases, not proof of where food was eaten or that advertising will work. '
                        'This is a promotion shortlist to review, with no online-channel inference.</p>'
                        + table_section(readable_dine_in(dine_in), ready=True))
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Combination validation review</title>
    <style>body{font:15px system-ui;margin:30px;color:#182b3a}table{border-collapse:collapse;font-size:13px}
    th,td{padding:10px;border:1px solid #ddd;text-align:left}th{background:#e9f1f7;position:sticky;top:0}
    tr:nth-child(even){background:#f7f9fb}h2{margin-top:36px}p{max-width:1000px}
    .table-scroll{overflow:auto}button,select{font:inherit;padding:9px 16px;margin:12px 0}
    button{cursor:pointer;background:#182b3a;color:white;border:0;border-radius:6px}
    [hidden]{display:none!important}</style></head><body>
    <h1>Combination validation review</h1><p>Exploratory candidates, not approved recommendations.
    Ranked within each scope using training support × gross profit. One unit of each product;
    historical median original prices and costs, without discounts. No test data was used.</p>
    <h2>What qualifies for inclusion?</h2>
    <p><b>Product and category combinations:</b> at least 20 purchases together in EACH period;
    bought together at least 20% more often than expected from individual popularity in BOTH periods;
    at least 5 in 100 orders with the first item also include the second in BOTH periods.
    Purchase rate and that conditional percentage must each retain at least 75% of their earlier level.
    Product combinations must also have positive original-price gross profit.</p>
    <p><b>Dine-in products:</b> at least 50 orders containing the product in EACH period,
    more than half dine-in, and a dine-in share at least 10 percentage points above the
    relevant branch's usual share in BOTH periods.</p>
    <p>These conservative starting rules define an owner-review shortlist, not proven business value.
    They are not significance tests and were introduced after exploring validation results;
    the untouched test period remains necessary for a final independent check.
    An empty shortlist is an acceptable result. Use “Show rejected candidates” to inspect failures.</p>
    <p>The tables below include all branches and an Overall view. Earlier orders found the patterns;
    later orders check whether they held up. Different period sizes mean purchase rates are more comparable than raw counts.
    These are buying associations, not proof one purchase causes another. Category patterns concern distinct categories only.</p>'''
    script = '''<script>
    document.querySelectorAll('.results').forEach(section => {
      const select = section.querySelector('select');
      const rows = [...section.querySelectorAll('tbody tr')];
      const more = section.querySelector('.more');
      const less = section.querySelector('.less');
      const rejected = section.querySelector('.rejected');
      let limit = 10;
      function render() {
        const matching = rows.filter(row => row.cells[0].textContent.trim() === select.value
          && (rejected.checked || row.cells[row.cells.length-1].textContent.trim() === 'Include for owner review'))
          .sort((a,b) => Number(a.cells[1].textContent) - Number(b.cells[1].textContent));
        rows.forEach(row => row.hidden = true);
        matching.slice(0,limit).forEach(row => row.hidden = false);
        section.querySelector('.count').textContent =
          `Showing ${Math.min(limit,matching.length)} of ${matching.length} candidates for ${select.value}. Ranking is based on training data; review the validation assessment before deciding.`;
        more.hidden = limit >= matching.length;
        more.textContent = `View ${Math.min(10, Math.max(0,matching.length-limit))} more`;
        less.hidden = limit <= 10;
      }
      select.addEventListener('change', () => {limit = 10; render();});
      rejected.addEventListener('change', () => {limit = 10; render();});
      more.addEventListener('click', () => {limit += 10; render();});
      less.addEventListener('click', () => {limit = 10; render();});
      render();
    });
    </script>'''
    (output / 'report.html').write_text(html + ''.join(sections) + script + '</body></html>', encoding='utf-8')


if __name__ == '__main__':
    main()
