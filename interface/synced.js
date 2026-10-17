// Adapters for the committed model contracts. No model fitting in the UI.
const forecastRows = () => data.forecasts.filter(r => branch==='Overall'||r.Branch===branch);
const branchRows = () => data.branch_predictions.filter(r => branch==='Overall'||r.Branch===branch);
forecast = function(){
 const rows=forecastRows(), dates=[...new Set(rows.map(r=>r.Date))].sort();
 const totals=dates.map(date=>rows.filter(r=>r.Date===date).reduce((s,r)=>s+r.expected_units,0));
 const max=Math.max(1,...totals), byProduct=new Map();
 rows.forEach(r=>byProduct.set(r.Product,(byProduct.get(r.Product)||0)+r.expected_units));
 const products=[...byProduct].sort((a,b)=>b[1]-a[1]);
 return title('Merchandise / Aya','Demand ahead.','14 days of expected unit sales.')+`<div class="grid"><div class="card"><span class="eyebrow">${esc(dates[0])}</span><strong class="metric">${num(totals[0])}</strong><span class="muted">Expected units · first forecast day</span></div><div class="card"><span class="eyebrow">14-day total</span><strong class="metric">${num(totals.reduce((a,b)=>a+b,0))}</strong><span class="muted">Expected units · ${esc(branch)}</span></div><div class="card"><span class="eyebrow">Pricing Optimization</span><strong class="metric">Pending</strong><span class="muted">No pricing output published</span></div></div><div class="card section"><h2>Daily outlook</h2><div class="chart" role="img" aria-label="Expected units per day">${dates.map((d,i)=>`<div class="column" style="height:${totals[i]/max*100}%" title="${d}: ${num(totals[i])} units"><span>${d.slice(8)}</span></div>`).join('')}</div><p class="notice">July 1–14, 2026 · Saved forecast, not a live forecast for today. Hover for units.</p></div><h2 class="section">Products · 14-day total</h2><div class="table-wrap"><table><thead><tr><th>Product</th><th>Expected units</th></tr></thead><tbody>${products.map(([p,n])=>`<tr><td>${esc(p)}</td><td>${num(n)}</td></tr>`).join('')}</tbody></table></div><p class="notice">Source: aya/outputs/forecast_next_14_days.csv. Revenue and optimized prices are not included in this output.</p>`;
};
branches = function(){
 const rows=branchRows().sort((a,b)=>Math.abs(b.cost_residual_pct)-Math.abs(a.cost_residual_pct));
 return title('Branch analysis model / YY','Price & cost signals.','Observed values beside model expectations.')+`<p class="notice">Test-period observations. Differences are statistical residuals, not recommended price changes or proven operational problems.</p><div class="table-wrap"><table><thead><tr><th>Branch / product</th><th>Unit cost</th><th>Expected cost</th><th>Cost difference</th><th>Listed price</th><th>Expected price</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.Product)}<br><small>${esc(r.Branch)}</small></td><td>${money(r.unit_cost)}</td><td>${r.prediction_complete?money(r.predicted_unit_cost):'Unavailable'}</td><td>${r.prediction_complete?num(r.cost_residual_pct)+'%':'Unavailable'}</td><td>${money(r.listed_unit_price)}</td><td>${r.prediction_complete?money(r.predicted_listed_unit_price):'Unavailable'}</td></tr>`).join('')}</tbody></table></div><p class="notice">Sorted by absolute cost difference. Source: branch-analysis-model/outputs/predictor_output.csv. No action thresholds are applied by this predictor.</p>`;
};
const originalDecisions=decisions;
decisions=function(){if(external.yehya)return originalDecisions();return title('Decision model / Yehya','The next move.','Combining marketing, demand and branch evidence.')+empty('Decision model in progress.','Action definitions are available. Generated decisions are not yet published.')+`<h2 class="section">Available action definitions</h2><div class="grid two">${data.decision_registry.map(r=>`<article class="card"><h3>${esc(r.label)}</h3>${badge(r.enabled?'Defined in registry':'Disabled')}<p class="muted">${esc(r.scopes.join(' · '))}</p></article>`).join('')}</div><p class="notice">These are supported action types, not model recommendations.</p>`};
overview=function(){return title('Stories / decision studio','Better decisions for every branch.','Use sales evidence to choose what to promote, what to prepare, and what to investigate.')+`
<div class="card"><span class="eyebrow">Why this exists</span><h2>Every choice affects the margin.</h2><p class="muted">A promotion needs a reason. Preparation needs a demand estimate. A price or cost difference needs a closer look.</p><p class="muted">This workspace brings that evidence together, so Stories can choose its next move.</p></div>
<div class="section"><h2>What you can explore</h2><div class="grid">
<a class="card" href="#combinations"><span class="eyebrow">Marketing</span><h2>What to promote</h2><p class="muted">Find products bought together and ideas for in-store promotion.</p><span>Explore pairings ↗</span></a>
<a class="card" href="#forecast"><span class="eyebrow">Merchandise</span><h2>What to prepare</h2><p class="muted">See expected product demand over the next 14 forecast days.</p><span>Explore demand ↗</span></a>
<a class="card" href="#branches"><span class="eyebrow">Branch analysis</span><h2>What to investigate</h2><p class="muted">Compare recorded prices and costs with model expectations.</p><span>Explore differences ↗</span></a>
</div></div>
<a class="card decision section" href="#decisions" style="display:block"><div class="row"><div><span class="eyebrow">Decisions</span><h2>Bring the evidence together.</h2><p class="muted">The decision model is being built to turn these signals into proposed actions.</p></div><span>View progress ↗</span></div></a>
<div class="section"><h2>Explore. Compare. Decide.</h2><p class="muted">Choose a branch. Inspect the evidence. Decide what deserves a small real-world test.</p></div>
<p class="notice">You stay in control. Patterns are not proof of impact, forecasts are estimates, and prices are never changed automatically. Final decisions and price optimization are still in development.</p>`};
const originalPairs=pairs;
pairs=function(){return originalPairs().replace('Mohammad / association analysis','Marketing Expert / association analysis')};
document.querySelector('nav a[href="#combinations"]').textContent='Marketing';
document.querySelector('nav a[href="#forecast"]').textContent='Merchandise';
document.querySelector('nav a[href="#branches"]').textContent='Branch analysis';
// Current published pricing scenarios, separate from the decision engine.
function pricingView(){
 const rows=data.pricing.filter(r=>branch==='Overall'||r.Branch===branch)
   .sort((a,b)=>b.profit_change_per_day_mid-a.profit_change_per_day_mid);
 const sum=field=>rows.reduce((s,r)=>s+r[field],0);
 return title('Pricing optimization','Price options.','Suggested prices. Assumed customer responses.')+
 `<div class="grid"><div class="card"><span class="eyebrow">Less price-sensitive</span><strong class="metric">${money(sum('profit_change_per_day_low'))}</strong><span class="muted">Estimated gross profit change / day</span></div><div class="card"><span class="eyebrow">Middle scenario</span><strong class="metric">${money(sum('profit_change_per_day_mid'))}</strong><span class="muted">Estimated gross profit change / day</span></div><div class="card"><span class="eyebrow">More price-sensitive</span><strong class="metric">${money(sum('profit_change_per_day_high'))}</strong><span class="muted">Estimated gross profit change / day</span></div></div>
 <p class="notice">Scenario estimates, not measured gains. Customer price sensitivity is assumed. Products are optimized separately; switching between products is not modeled.</p>
 <div class="table-wrap"><table><thead><tr><th>Product / branch</th><th>Current price</th><th>Suggested price</th><th>Change</th><th>Middle scenario / day</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.Product)}<br><small>${esc(r.Branch)}</small></td><td>${money(r.price)}</td><td>${money(r.new_price)}</td><td>${badge(r.action)} ${num(r.change_pct)}%</td><td>${money(r.profit_change_per_day_mid)} gross profit<br><small>${num(r.units_per_day_mid)} expected units</small></td></tr>`).join('')}</tbody></table></div><p class="notice">Source: PricingOptimization/outputs/price_recommendations.csv. Prices have not been applied. A controlled price test is needed.</p>`;
}
const forecastBeforePricing=forecast;
forecast=function(){return forecastBeforePricing()
 .replace('<strong class="metric">Pending</strong><span class="muted">No pricing output published</span>', '<strong class="metric">Available</strong><a href="#pricing">Explore price options ↗</a>')
 .replace('aya/outputs/forecast_next_14_days.csv','Merchandise/outputs/forecast_next_14_days.csv')
 .replace('Revenue and optimized prices are not included in this output.','Revenue is not included in this forecast. Pricing scenarios are available in Pricing.');};
const decisionsBeforeUpdate=decisions;
decisions=function(){return decisionsBeforeUpdate()
 .replace('Decision model in progress.','Decision engine · not trained')
 .replace('Action definitions are available. Generated decisions are not yet published.','Implementation is available. Training is disabled; a validated action-response simulator is still needed.')};
const homeBeforeUpdate=overview;
overview=function(){return homeBeforeUpdate()
 .replace('The decision model is being built to turn these signals into proposed actions.','The decision engine is implemented. Training and real-world action testing are still pending.')
 .replace('Final decisions and price optimization are still in development.','Pricing scenarios are available for review. Final decision outputs are still pending.')
 + `<a class="card section" href="#pricing" style="display:block"><span class="eyebrow">Pricing</span><h2>Explore a price change.</h2><p class="muted">Compare suggested prices and estimated profit across three customer-response scenarios.</p><span>View price options ↗</span></a>`;};
const pricingLink=document.createElement('a');pricingLink.href='#pricing';pricingLink.textContent='Pricing';document.querySelector('nav').append(pricingLink);
const baseRender=render;
render=function(){if(location.hash==='#pricing'){$('#breadcrumb').textContent='Studio / Pricing';document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.hash==='#pricing'));$('#content').innerHTML=pricingView();$('#period').textContent=data.period;}else baseRender();};
window.onhashchange=render;
render();
