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
const manualDecisions=decisions;
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
let decisionSearch='',decisionOffset=0;
const decisionPageSize=6;
const committedDecisionOutput=()=>data.decision_output||null;
const allDecisionActions=()=>committedDecisionOutput()?.synthetic_demo?.actions||[];
function matchingDecisionActions(){
 const query=decisionSearch.trim().toLowerCase();
 return allDecisionActions().filter(item=>(branch==='Overall'||item.branch===branch)&&(!query||[item.title,item.branch,...(item.products||[]),item.explanation].join(' ').toLowerCase().includes(query)));
}
function decisionCard(item){
 const impact=item.expected_impact==null?'':`<p><strong>Expected impact</strong> · ${esc(item.expected_impact)}</p>`;
 const period=item.period==null?'':`<p class="muted">Period · ${esc(item.period)}</p>`;
 const source=(item.supporting_evidence||[]).map(e=>e.source).filter(Boolean).join(' · ');
 const assumption=(item.assumptions||[])[0];
 return `<article class="card decision-result"><div class="row"><span class="badge simulated">Simulated review proposal</span><span class="muted">${esc(item.branch)}</span></div><h2>${esc(item.title)}</h2><p class="decision-target">${esc((item.products||[]).join(' + ')||'No product supplied')}</p><p>${esc(item.explanation||'Explanation not supplied.')}</p>${impact}${period}<p class="muted">Evidence · ${esc(source||'Source not supplied')}</p>${assumption?`<p class="assumption"><strong>Important assumption</strong> · ${esc(assumption)}</p>`:''}<p><strong>Next step</strong> · ${esc(item.next_step||'Not supplied')}</p><button data-decision-detail="${esc(item.id)}">Details</button></article>`;
}
decisions=function(){
 const output=committedDecisionOutput();
 if(!output?.available){return external.yehya?manualDecisions():title('Decision model','Decision results.','Proposals for management review.')+empty('No decision output found.','Run the decision model and refresh the website data.');}
 const real=output.real_evaluation||{},synthetic=output.synthetic_demo||{},rows=matchingDecisionActions();
 if(decisionOffset>=rows.length)decisionOffset=0;
 const pageRows=rows.slice(decisionOffset,decisionOffset+decisionPageSize);
 const realCount=(real.actions||[]).length,status=real.recommended==='stop'?'STOP':real.recommended||'Not supplied';
 const simulatedDelta=synthetic.simulated_net_incremental_profit_usd;
 return title('Decision model','Review the proposed checks.','Model-selected reviews only. Nothing is applied automatically.')+
 `<div class="grid decision-summary"><div class="card"><span class="eyebrow">Frozen real evaluation</span><strong class="metric">${esc(status)}</strong><span class="muted">${realCount} validated interventions</span></div><div class="card"><span class="eyebrow">Synthetic sandbox</span><strong class="metric">${num(synthetic.action_count||0)}</strong><span class="muted">Review proposals · assumed effects</span></div><div class="card"><span class="eyebrow">Automatic changes</span><strong class="metric">0</strong><span class="muted">Review only</span></div></div>`+
 `<p class="notice decision-warning"><strong>Measured result:</strong> the real policy stopped because no validated action-response model is available. <strong>Simulated result:</strong> ${simulatedDelta==null?'financial impact was not supplied':`${money(simulatedDelta)} modeled net change`}; this is not observed uplift.</p>`+
 `<div class="controls decision-tools"><label class="sr-only" for="decision-search">Search proposals</label><input id="decision-search" type="search" placeholder="Find a branch or product…" value="${esc(decisionSearch)}"><span class="muted">${rows.length} result${rows.length===1?'':'s'} · ${esc(branch)}</span></div>`+
 (pageRows.length?`<div class="grid two decision-grid">${pageRows.map(decisionCard).join('')}</div>`:`<div class="empty"><h2>No decision results for this view.</h2><p>Try another branch or clear the search.</p></div>`)+
 `<div class="row controls decision-pager"><span class="muted">${rows.length?`${decisionOffset+1}–${Math.min(decisionOffset+decisionPageSize,rows.length)} of ${rows.length}`:'No matching results'}</span><div><button id="decision-prev" ${decisionOffset===0?'disabled':''}>Previous</button> <button id="decision-next" ${decisionOffset+decisionPageSize>=rows.length?'disabled':''}>Next</button></div></div>`+
 `<p class="notice">Source: ${esc(output.artifact_sources?.synthetic_demo||'decision model output')}. Detailed evidence and guardrails are available under Details.</p>`;
};
function showDecisionDetail(id){
 const item=allDecisionActions().find(row=>row.id===id);if(!item)return;
 const raw=item.raw_evidence||{},observed=raw.observation||{},expected=raw.model_expectation||{},residuals=raw.residual_signals||{};
 const evidenceRows=[...Object.entries(observed),...Object.entries(expected),...Object.entries(residuals)];
 $('#detail-content').innerHTML=`<div class="eyebrow">${esc(item.branch)} · Decision model</div><h2>${esc(item.title)}</h2><span class="badge simulated">Synthetic review proposal</span><dl><dt>Affected products</dt><dd>${esc((item.products||[]).join(' + ')||'Not supplied')}</dd><dt>Why it was selected</dt><dd>${esc(item.explanation||'Not supplied')}</dd><dt>Supporting evidence</dt><dd>${evidenceRows.length?evidenceRows.map(([key,value])=>`${esc(key.replaceAll('_',' '))}: ${typeof value==='number'?num(value):esc(value)}`).join('<br>'):'Not supplied'}</dd><dt>Expected impact and period</dt><dd>${item.expected_impact==null&&item.period==null?'Not supplied by the model.':`${item.expected_impact==null?'':esc(item.expected_impact)} ${item.period==null?'':`· ${esc(item.period)}`}`}</dd><dt>Important assumptions</dt><dd>${(item.assumptions||[]).length?item.assumptions.map(esc).join('<br>'):'Not supplied'}</dd><dt>Suggested next step</dt><dd>${esc(item.next_step||'Not supplied')}</dd><dt>Safety</dt><dd>This result is for review only. It does not apply a price, cost, or operational change.</dd></dl>`;
 $('#detail').showModal();
}
overview=function(){
 const output=committedDecisionOutput(),real=output?.real_evaluation||{},synthetic=output?.synthetic_demo||{};
 const status=real.recommended==='stop'?'STOP':real.recommended||'Unavailable';
 return title('Stories / decision studio','Better decisions for every branch.','Review measured evidence and clearly labeled simulations before choosing a test.')+`
 <div class="grid"><div class="card"><span class="eyebrow">Real decision status</span><strong class="metric">${esc(status)}</strong><span class="muted">No validated intervention response</span></div><div class="card"><span class="eyebrow">Simulated reviews</span><strong class="metric">${num(synthetic.action_count||0)}</strong><span class="muted">Assumption-based proposals</span></div><div class="card"><span class="eyebrow">Applied changes</span><strong class="metric">0</strong><span class="muted">Management review required</span></div></div>
 <div class="section"><h2>What you can explore</h2><div class="grid"><a class="card" href="#combinations"><span class="eyebrow">Marketing</span><h2>What to promote</h2><p class="muted">Products bought together and in-store patterns.</p><span>Explore pairings →</span></a><a class="card" href="#forecast"><span class="eyebrow">Merchandise</span><h2>What to prepare</h2><p class="muted">Expected product demand over the next 14 forecast days.</p><span>Explore demand →</span></a><a class="card" href="#branches"><span class="eyebrow">Branch analysis</span><h2>What to investigate</h2><p class="muted">Recorded prices and costs beside model expectations.</p><span>Explore differences →</span></a></div></div>
 <a class="card decision section" href="#decisions" style="display:block"><div class="row"><div><span class="eyebrow">Decision output</span><h2>Review model-selected checks.</h2><p class="muted">The frozen real policy stopped. Synthetic review proposals are available with assumptions and guardrails.</p></div><span>Review results →</span></div></a>
 <a class="card section" href="#pricing" style="display:block"><span class="eyebrow">Pricing</span><h2>Explore a price scenario.</h2><p class="muted">Compare assumed customer responses. These are not measured gains.</p><span>View price options →</span></a>
 <p class="notice">Nothing here changes prices or operations. Real benefits require a validated response model or controlled experiment.</p>`;
};
const pricingLink=document.createElement('a');pricingLink.href='#pricing';pricingLink.textContent='Pricing';document.querySelector('nav').append(pricingLink);
const baseRender=render;
render=function(){if(location.hash==='#pricing'){$('#breadcrumb').textContent='Studio / Pricing';document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.hash==='#pricing'));$('#content').innerHTML=pricingView();$('#period').textContent=data.period;}else baseRender();};
window.onhashchange=render;
render();
