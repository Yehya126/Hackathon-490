// Human-reviewed shortlist from committed outputs; NOT the PPO model's decisions.
const manualDecisions = [
 {branch:'Garden',title:'Test one sandwich price before a wider change',kind:'Small price pilot',
  evidence:'Chicken Pesto Sandwich: $10.75 → $11.80. Estimated extra gross profit: $0.98–$2.11/day across the published scenarios; middle scenario $1.62/day.',
  context:'Merchandise forecasts 38.31 units for July 1–14 at the original conditions. That forecast is not a prediction at the proposed price.',
  next:'Use a small controlled price test. Track gross profit per eligible order, units and substitution. Confirm current costs and menu price first.',
  limit:'Price sensitivity is assumed, not measured. Do not roll out all 34 Garden price rises together.',
  sources:['PricingOptimization/outputs/price_recommendations.csv — Garden / Chicken Pesto Sandwich','Merchandise/outputs/forecast_next_14_days.csv — July 1–14, 2026']},
 {branch:'Garden',title:'Review Garden’s cost structure',kind:'Investigate',
  evidence:'The branch model estimates Harbour cost at 0.699× Garden and Uptown at 0.671× Garden after product effects. Garden’s cost level is therefore about 43% and 49% higher, respectively.',
  context:'Garden also accounts for $25.26/day of the pricing model’s $35.50/day middle-scenario gain. This supports prioritizing a cost review, not assuming costs can be reduced.',
  next:'Check supplier invoices, portions, product specifications and accounting allocation for matched products.',
  limit:'A branch effect is not proof of waste or overcharging. Small product residuals do not establish an operational fault.',
  sources:['branch-analysis-model/outputs/decision_model_input.json — model.branch_multipliers.cost','PricingOptimization/outputs/summary.json — expected_profit_change']},
 {branch:'Harbour',title:'Try an in-store Tiramisu placement',kind:'Promotion pilot',
  evidence:'265 of 398 Tiramisu orders were dine-in: 66.6%, versus the branch’s usual 59.0%. About 44 dine-in orders per month.',
  context:'This is a placement idea at the existing price. No extra revenue or profit has been established.',
  next:'Compare display and no-display periods with a randomized, balanced schedule. Track gross profit per eligible order and dessert substitution.',
  limit:'Historical preference may reflect customer mix. No discount is justified by this evidence alone.',
  sources:['Marketing_Expert/artifacts/full_data/dine_in.csv — Harbour / Tiramisu Cup']},
 {branch:'Harbour',title:'Test a pairing without discounting it',kind:'Tentative pairing',
  evidence:'Cinnamon Roll + Strawberry Matcha: 38 shared orders, about 6.3/month. Original-price total $13.44; estimated gross profit $8.02. Association is 23.5% above individual popularity.',
  context:'This is a low-volume exploratory pattern, not proof customers will accept a bundle.',
  next:'Try a small menu-placement experiment at original prices. Record incremental basket profit, not just combo sales.',
  limit:'The marketing model used all historical data. No independent validation or causal promotion effect is available.',
  sources:['Marketing_Expert/artifacts/full_data/products.csv — Harbour / Cinnamon Roll + Strawberry Matcha']},
 {branch:'Uptown',title:'Feature Coffee Frappe inside the shop',kind:'Promotion pilot',
  evidence:'123 of 170 Coffee Frappe orders were dine-in: 72.4%, versus the branch’s usual 60.3%. About 20.5 dine-in orders per month.',
  context:'The 12.1-point difference supports a small in-store placement test. It does not support an online-advertising conclusion.',
  next:'Test a counter or table display at the existing price. Measure eligible-order gross profit and changes in other cold-drink sales.',
  limit:'The pattern is observational. Keep the pilot inexpensive; advertising response is unknown.',
  sources:['Marketing_Expert/artifacts/full_data/dine_in.csv — Uptown / Coffee Frappe']},
 {branch:'Overall',title:'Refresh forecasts before planning stock',kind:'Required before use',
  evidence:'The saved July 1–14 forecast totals 5,370.82 units: Garden 1,742.59; Harbour 2,624.09; Uptown 1,004.14.',
  context:'Those dates are historical. Do not use this export as tomorrow’s preparation list or add price-scenario effects to it.',
  next:'Get recent transactions and rerun the forecast for the actual planning dates. Adjust preparation for stock, shelf life and capacity.',
  limit:'Unit forecasts are estimates, not purchase orders. Revenue and action-conditioned demand are not supplied by this export.',
  sources:['Merchandise/outputs/forecast_next_14_days.csv — summed expected_units by branch']},
 {branch:'Harbour',title:'Check the historical Matcha sales drop',kind:'Historical follow-up',
  evidence:'On June 9, Matcha & Tea sold 22 units against 40.112 expected. The model flagged an unusual drop.',
  context:'The reported roughly $79 profit difference is relative to a model expectation—not a confirmed recoverable loss.',
  next:'Check stockouts, opening hours and transaction capture for that day before assigning a cause.',
  limit:'This is a historical alert, not evidence of a current incident.',
  sources:['Merchandise/outputs/alerts_test.csv — 2026-06-09 / Harbour / Matcha & Tea']}
];
decisions=function(){
 if(external.yehya)return originalDecisions();
 const rows=manualDecisions.filter(r=>branch==='Overall'||r.branch===branch||r.branch==='Overall');
 return title('Manual review / all model outputs','Actions worth testing.','A reviewed shortlist. Not output from the decision engine.')+
 `<p class="notice">No actions have been applied. Evidence spans different historical periods. Benefits are unproven or scenario-based. The PPO decision engine remains untrained.</p>`+
 rows.map(r=>`<article class="card decision section"><div class="row">${badge(r.kind)}<span class="muted">${esc(r.branch)}</span></div><h2>${esc(r.title)}</h2><p>${esc(r.evidence)}</p><p class="muted">${esc(r.context)}</p><h3>Next step</h3><p>${esc(r.next)}</p><details><summary>Evidence & limits</summary><p>${esc(r.limit)}</p><ul>${r.sources.map(s=>`<li class="muted">${esc(s)}</li>`).join('')}</ul></details></article>`).join('')+
 `<p class="notice">Manual shortlist based on exported results. Do not sum suggested benefits: actions can overlap or change one another’s effects. Imported Yehya decisions replace this temporary view.</p>`;
};
render();
