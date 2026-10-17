# Stories Coffee — Continuous Business Intelligence & Recommendation System

> **Project reference / implementation blueprint**  
> Status: ideation and design; models and recommendation policies not yet implemented.  
> Dataset: `stories_transactions.csv` (January–June 2026).  
> Purpose: continuously identify meaningful business problems and opportunities as transaction data arrives, explain the evidence, and recommend actions for management.

## 1. Vision

Build a **multi-specialist business intelligence and recommendation engine**, not a one-off exploratory dashboard or a chatbot that answers database questions.

The system should routinely answer:

1. **What is wrong or unusually different?** Across branches and within each branch over time.
2. **Why is it happening?** Separate observed effects of cost, selling price, discounts, volume, and product mix.
3. **What should Stories do?** Recommend specific investigations, operational changes, or controlled experiments.
4. **What is the financial significance?** Estimate *observed exposure* or *conditional scenario value*, never pretend an untested action has a proven payoff.
5. **Is the recommendation still relevant?** Re-evaluate and suppress duplicate alerts as fresh data arrives.

**Design principle:** Every specialist produces **finding → evidence → explanation → actionable recommendation → confidence/limitation**. Forecasts or plots without a decision implication are supporting tools, not the deliverable.

## 2. Dataset and verified constraints

| Property | Verified value / implication |
|---|---|
| Time span | 2026-01-01 to 2026-06-30 |
| Rows | 51,177 **product-line records** |
| Columns | 14 |
| Unique transactions | 33,887 |
| Branches | Harbour, Garden, Uptown |
| Products and categories | 45 products, 9 categories |
| Recorded revenue | $305,925.14 |
| Recorded gross profit | $179,430.49 |
| Missing customer IDs | ~36.5% of product-line records |
| Price history | For all 135 branch–product pairs, recorded **listed unit price is constant** over the six months |
| Cost history | For all 135 branch–product pairs, recorded **unit cost is constant** over the six months |
| What does vary | Transactions, quantities, time, branch/category/product mix, discount occurrence/rate, revenue and gross profit |

**Exact CSV columns**

```text
Transaction ID
Date
Time
Branch
Customer ID
Order Type
Product
Category
Quantity
Unit Price (USD)
Discount Rate
Unit Cost (USD)
Line Revenue (USD)
Line Gross Profit (USD)
```

**Data grain matters:** A CSV row is a product line, **not an order**. A `Transaction ID` can occur on multiple rows. Compute order counts and average order value on order-level aggregates; compute product units and product contribution on line-level aggregates. Do not deduplicate valid multi-line orders.

**Critical non-goals from existing data:**

- Do **not** propose a historical unit-cost increase detector or cost forecasting model on this six-month dataset: there is no within-branch/product unit-cost variation to learn from.
- Do **not** claim that branch price differences identify price elasticity, optimal prices, or causal responses to price changes; each branch–product price is fixed.
- Do **not** infer actual net profit: the dataset gives product-level **gross profit**, not labor, rent, overhead, waste, or total operating cost.
- The regularity of branch-specific pricing/costing may reflect rules used to create or record the data. **Validate the source accounting rules** before treating cross-branch discrepancies as operational inefficiencies.
- Customer-level conclusions must address missing customer IDs and potential coverage bias.

## 3. System architecture

```text
               NEW TRANSACTION DATA (daily / appended CSV)
                                |
                                v
                  Ingest, validate, standardize
                                |
             +------------------+------------------+
             |                  |                  |
       Order-level data   Product-line data   Daily/weekly aggregates
             |                  |                  |
             +------------------+------------------+
                                |
              +-----------------+-----------------+
              |                 |                 |
        [1] CEO Analyst   [2] Branch Monitor  [3] Pricing Analyst
           Cross-branch       Within-branch         Price/promo
              |                 |                 |
              +----------+------+-----------------+
                         |      |
                    [4] Basket Analyst
                         |      |
                         +------+
                            |
                 [5] Recommendation Prioritizer
                            |
                  Evidence + ranked action cards
                            |
                   CEO / branch dashboard
                            |
                   Feedback: act / defer /
                   dismiss / resolve / outcome
                            |
                   Re-evaluate as data arrives
```

These are **specialized logical modules**, not necessarily five deep neural networks. Use a trained model where predictions or learned patterns are justified; use deterministic accounting calculations where answers are exactly computable; use statistical tests for deviations. The overall result is a **multi-model / multi-method decision-support system**.

---

## 4. Specialist 1 — CEO Analyst: Cross-Branch Intelligence

**Question:** Which branches are unusually expensive, underpriced, low-margin, or selling an unfavorable product mix relative to comparable branches?

### Inputs

- Branch, product, category, time window.
- Unit price, unit cost, units sold, discounts.
- Line revenue, line gross profit.
- Derived metrics: realized revenue per unit, cost per unit, gross profit per unit, gross margin, category/product sales mix, contribution to total gross profit.
- Comparable products present in multiple branches.

### Detection and explanation

1. Build **matched-product** comparisons, holding the product definition constant.
2. Identify branches with unusual price, cost, and margin differences.
3. Check whether differences are **systematic across many products** or localized to a few SKUs/categories.
4. **Standardize the product mix** to avoid confusing category composition with differences in economics.
5. Decompose the margin gap into **unit-cost effects, listed-price effects, discounts, and mix effects**, using clearly defined counterfactual ordering/attribution rules.
6. Rank the products/categories responsible for the largest *recorded* economic exposure.

**Initial methods:** matched-SKU arithmetic, robust cross-branch comparisons, decomposition, percentile/outlier thresholds. Isolation Forest or another anomaly model is optional if there are enough meaningful observations; with only three branches and 135 matched branch–product records, a complex model may be less defensible than transparent calculations.

### Output

- Branch(es), affected SKUs/categories, detected discrepancy, supporting comparisons.
- Root-factor attribution (e.g., high recorded cost rather than bad product mix).
- **Conditional financial scenario**, with the benchmark and assumptions stated.
- Recommended investigation, ordered by affected products and opportunity size.
- One persistent issue card rather than a repeated notification every day for a stable gap.

### Verified starting case: Garden

- Garden revenue: **$99,748.55**; gross profit: **$46,794.49**; gross margin: **46.91%**.
- Harbour gross margin: **64.96%**; Uptown: **62.63%**.
- For all **45 matched products**, Garden's recorded unit cost is higher than Harbour's (median ratio roughly **+43%**); Garden's listed price is lower (median difference roughly **−4.7%**).
- Holding Garden's quantities and discount rates fixed, replacing its unit costs with Harbour's matched-product costs gives a **~$15,923** higher six-month gross profit, to **~$62,718**.
- Also replacing Garden's listed prices with Harbour's at fixed quantities gives **~$67,704** gross profit in the scenario.

**Proposed action:** audit branch costing/pricing rules, supplier contracts, recipe definitions, and specific SKUs contributing most to the discrepancy. **Do not** claim $15.9k is guaranteed achievable savings or that changing prices leaves demand unchanged.

### Charts

- Branch margin and cost-of-goods comparison.
- Matched-SKU price and cost scatter plots (reference line = equality).
- Cost/price/discount/mix **profit-gap waterfall**.
- SKU/category **Pareto chart** of modeled cost discrepancies.
- Branch-level gross margin with identical category weights.

---

## 5. Specialist 2 — Branch Monitor: Within-Branch Performance

**Question:** What has changed inside a given branch, is the change unusual, and which observed factors explain it?

### Inputs

- Branch, date, hour, weekday, month, order type.
- Transaction counts, order value, units, category/product quantities.
- Revenue, gross profit, margin, discount activity, product-mix shares.
- Lagged and rolling historical summaries calculated **only from preceding dates**.

### Detection and explanation

1. Learn expected revenue, units, transactions and/or gross profit for a branch/category/day or hour (use seasonal baselines first, then test gradient boosting such as LightGBM/XGBoost).
2. Compare actual performance with prediction intervals / calibrated residual thresholds.
3. When abnormal, **attribute** gross-profit differences to order volume, units per order, changing product mix, and discount activity. In this dataset, unit prices and unit costs are constant within branch–product pairs, so do not invent cost shocks.
4. Contrast the change with similar weekdays, hours and recent weeks; track sustained changes separately from one-day noise.
5. Distinguish an **unusual business outcome** from a **bad data record**. Validate arithmetic and schema separately.

For any period \(t\), the recorded economics can be represented as:

```text
Gross profit(t) = sum over product lines [
    Quantity * (Unit Price * (1 - Discount Rate) - Unit Cost)
]
```

Use the actual line-level revenue/profit fields for financial reporting and reconcile against the formula with an appropriate rounding tolerance. For exact attribution, specify a consistent decomposition convention, because mix, discounts, and volume interact.

### Output

- Alert: expected vs. actual metric, absolute and relative deviation, uncertainty, persistence.
- Contributing product categories, hours, order types and discount activity.
- Suggested operational check: verify promotion, investigate demand shift, adjust preparation, review product availability, etc.
- State whether the finding is a **diagnostic fact** or an **untested hypothesis**.

### Example (illustrative; not an observed Garden result)

> “Weekly gross profit is 15% below its expected baseline. Most of the accounting difference is associated with fewer transactions and a shift toward lower-margin categories. Inspect these categories and compare comparable weeks before changing operations.”

### Charts

- Actual vs. expected gross profit/revenue with prediction bands.
- Daily gross-profit anomaly timeline; normalized by weekday.
- Branch × weekday × hour demand heatmaps (separate dine-in and takeaway).
- Revenue/gross-profit change attribution waterfall.
- Product-category mix over time and discounts over time.

---

## 6. Specialist 3 — Pricing & Promotion Analyst

**Question:** Where might a price, discount, or promotional policy deserve a controlled test?

### Inputs

- Product/branch unit prices and costs.
- Quantity, gross margin, discount rate and discounted volume.
- Historical product/category sales, weekday/hour mix, order type.
- Matched-product pricing across branches.

### Detection and explanation

1. Flag unusually low-margin products or discrepancies between equivalent products across branches.
2. Compute **scenario economics** for proposed price/discount changes with explicit assumptions (e.g., unchanged units sold).
3. Assess recorded promotion periods: compare category units, revenue and gross profit to comparable time windows and branches, controlling where feasible for weekday, season and product mix.
4. Recommend **experiments** for candidates with good business rationale; do not infer a causal lift or price elasticity from fixed-price observational data.
5. Once deliberate price/promotion experiments are logged, optionally train a response/uplift model.

### Output

- Candidate SKU and branch; what pricing/promotion policy is under review.
- Current price, cost, margin; comparison branch; scenario range and its assumptions.
- Recommended controlled experiment, control group, measurement window, guardrails.
- Evaluation target: **incremental gross profit**, not revenue alone.

### Existing hypothesis to examine

Harbour shows a distinctive March/May **15% discount pattern** on Cold Coffee and Desserts. That is a promising promotion-analysis test case, **not proof** that the promotion increased or decreased sales.

### Charts

- Price vs. gross-margin scatter, matched by product across branches.
- Promotion-event timeline with category units and gross profit.
- Discounted vs. comparison series (with controls and explicit caveats).
- Sensitivity chart: profit under different assumed volume responses to a price change.

---

## 7. Specialist 4 — Basket & Product Intelligence

**Question:** Which product combinations are meaningful, and what cross-sell or merchandising experiments are worth trying?

### Inputs

- Transaction ID; products and quantities within the same order.
- Branch, time, order type and optional usable customer ID.
- Product category, price, unit cost and gross profit contribution.

### Detection and explanation

1. Construct baskets by **distinct Transaction ID** (not by individual rows).
2. Mine co-purchases via **FP-Growth / association rules**.
3. Require minimum sample counts/support; rank by lift, confidence and repeatability across periods/branches.
4. Include product gross margin and order context when recommending pairs or add-ons.
5. Watch for a difference between co-purchase association and incremental demand: a common pair is **not** proof a bundle raises profit.

### Output

- Product A → candidate B, branch/order-type/time context.
- Support, confidence, lift, sample count, margin information.
- Recommendation to **test** a checkout suggestion, display placement or bundle; specify an A/B test to measure incremental gross profit.

### Charts

- Product association lift/support heatmap.
- Basket size and gross-profit distribution by branch/order type.
- Co-purchase network filtered by significance and practical margin.
- Candidate add-on ranking with coverage and margin.

---

## 8. Specialist 5 — Recommendation Prioritizer & Feedback

**Question:** Of all findings from the four specialists, which deserve management attention first?

### Inputs

Normalized findings from every specialist:

- Finding type, affected branch/product/category/time window.
- Evidence, statistical score, observed exposure or scenario estimate.
- Confidence and data limitations.
- Proposed action, action type (investigate / operate / experiment).
- Historic recommendation status and management feedback.

### Ranking policy (initially transparent, not a mystery score)

- **Financial materiality:** measured loss/exposure or bounded conditional scenario, clearly labeled.
- **Evidence strength:** stable comparisons, sample size, predictive calibration or statistical support.
- **Actionability:** can management verify or test it at reasonable effort?
- **Novelty/persistence:** do not send the same unresolved static finding daily.
- **Risk:** discount cost, service quality, likely implementation difficulty.

Initially, use a documented, weighted rule-based score, with **separate ranks for verified discrepancies and speculative opportunities**. Later, management feedback and observed experimental outcomes can train a ranking model.

### Recommended record schema

```json
{
  "finding_id": "stable-deduplicated-key",
  "specialist": "ceo_analyst",
  "scope": {"branch": "Garden", "category": null, "product": null},
  "period": {"start": "2026-01-01", "end": "2026-06-30"},
  "problem": "Recorded product costs systematically exceed Harbour's",
  "evidence": ["45/45 matched products higher in Garden"],
  "impact": {
    "kind": "fixed_volume_counterfactual",
    "gross_profit_usd": 15923.47,
    "assumptions": "Garden quantities/discounts unchanged; Harbour SKU costs substituted"
  },
  "recommendation": "Verify costing rules and investigate high-exposure SKUs",
  "action_type": "investigate",
  "confidence": "high_descriptive_evidence; causal_explanation_unverified",
  "status": "new"
}
```

Recommendation state: `new` → `acknowledged` → `testing` / `dismissed` → `resolved`. Alert reappearance should depend on meaningful new evidence, not just a daily rerun.

---

## 9. Updating with new data

Suggested daily or weekly pipeline:

1. **Ingest** fresh rows. Avoid reprocessing duplicate uploads while retaining legitimate repeated product lines within a single order. Ideally add a stable source row ID or ingestion batch + row index.
2. **Validate** schema, parse timestamps, type-check numerics, enforce sensible ranges, reconcile provided line revenue and gross profit, flag outliers/missingness.
3. **Update features:** line, transaction, day-hour, branch-category, and matched-product views.
4. **Run specialists** with an explicit historical cutoff. Retrain forecasting models on a schedule or after measured distribution drift; recompute deterministic economics on each update.
5. **Generate findings**, including baseline comparisons, uncertainty and scenario assumptions.
6. **Deduplicate and rank**, preserving recommendation history.
7. **Publish dashboard** and human-reviewable evidence.
8. **Record feedback and experimental outcomes** for future learning.

**Important:** Existing static branch-to-branch cost differences can be discovered on day one, but should not trigger repeated “costs increased” notifications. If future data really introduces cost or price changes, explicitly detect the new variation and extend the model then.

## 10. Evaluation / Hackathon demonstration

The system must be demonstrably better than a generic summary dashboard or generic language-model suggestions.

| Specialist | First baseline | Evaluation |
|---|---|---|
| CEO Analyst | Simple branch ranking by revenue or gross margin | Detect/diagnose planted or held-out matched-SKU discrepancies; attribution correctness; value captured in top-K findings |
| Branch Monitor | Previous-week / weekday historical mean and fixed percentage alerts | Time-ordered forecast MAE/WAPE; anomaly precision, recall and false alerts on held-out or explicitly injected changes |
| Pricing Analyst | Static minimum-margin alerts or promotional sales rankings | Accuracy of accounting scenarios; experiment selection quality; **real** incremental gross profit only when experiments exist |
| Basket Analyst | Top-selling products / most common pairs | Association stability and holdout lift; real cross-sell lift only with intervention tests |
| Prioritizer | Sort alerts by revenue impact alone | Top-K relevance rated by managers; duplicate suppression; eventual verified business outcomes |

**Data split:** Use a chronological split, such as January–April for initial training, May for validation/calibration, and June for a realistic streaming replay. Do not let rolling features contain future observations. Report results per branch and category, not only an overall average.

**Synthetic anomaly injections:** acceptable to test detection mechanics, but must be labeled *injected/constructed*, not real business events. Do not claim uplift or ROI without interventions.

## 11. Implementation order

### Phase 1 — Business evidence and metrics (highest priority)

- [ ] Fix the line-level and order-level data model.
- [ ] Create matched-product price/cost/margin comparisons.
- [ ] Reproduce Garden's cross-branch discrepancy **without branch-specific hardcoding**.
- [ ] Implement fixed-volume counterfactuals and identify highest-exposure SKUs.
- [ ] Implement persistent recommendation identifiers and clear caveat labels.

### Phase 2 — Within-branch monitoring

- [ ] Aggregate daily/weekly/hourly performance with appropriate weekday normalization.
- [ ] Set a simple seasonal baseline, then test a learned forecasting model.
- [ ] Add outlier alerts plus exact accounting decomposition of performance shifts.
- [ ] Show actual versus expected, attribution and actionable next steps.

### Phase 3 — New recommendation classes

- [ ] Build pricing/promotion scenario and experimental-test recommendations.
- [ ] Mine transaction baskets and rank meaningful, higher-margin cross-sell opportunities.
- [ ] Route all findings through a common prioritizer.

### Phase 4 — Continuous operation and validation

- [ ] Replay held-out months as though data is arriving incrementally.
- [ ] Compare each specialist against simple baselines.
- [ ] Add manual management feedback and recommendation lifecycle.
- [ ] Package a dashboard that displays the **best new recommendations**, not an endless list of plots.

## 12. Demo story

1. Load the first months of transactions; the system autonomously identifies the **Garden cross-branch cost/margin anomaly** and quantifies the *conditional* opportunity.
2. Feed additional weeks/months; the Branch Monitor identifies whether observed gross-profit shifts are unusual **relative to each branch's own calendar patterns** and explains their accounting drivers.
3. Show a pricing or basket opportunity with the proposed experiment and its measurement plan.
4. Present ranked recommendation cards with impact type, evidence, limitations and action status.
5. Demonstrate that a known persistent issue is not rediscovered as a “new anomaly” with every upload.

## 13. Explicit assumptions and open questions

- Are branch-specific prices and costs operational facts, accounting allocations, or synthetic rules?
- Is `Unit Cost (USD)` an ingredient cost, a fully loaded product cost, or something else?
- Do we have future data with real price/cost changes, stockouts, promotions, staff schedules, or supplier invoices?
- What actions can the CEO actually take (prices, suppliers, scheduling, merchandising), and at what cadence?
- What business guardrails matter (minimum margin, availability, price consistency, customer experience)?
- Can we log management decisions and execute real controlled trials? Without intervention feedback, “recommendations improve profits” remains a hypothesis.

---

## Bottom line

Build a **continuous, evidence-driven recommendation system** powered by multiple specialized analytical models. The initial centerpiece is **cross-branch profit diagnosis**, complemented by **within-branch change detection**, **pricing/promotion experiment selection**, **basket opportunities**, and **recommendation prioritization**. Garden is the first discovered test case—not a rule programmed into the system or the only business problem it can ever find.
