# Codex Implementation Prompt — Stories Cross-Branch Profit Intelligence Engine

## 1. Project Objective

Build a Python-based **Cross-Branch Profit Intelligence Engine** for Stories Coffee using `stories_transactions.csv`.

The system must:

1. Automatically detect abnormal cost, price, discount, and profitability differences between branches.
2. Learn systematic branch-level and product-level differences using statistical models.
3. Explain profitability gaps through financial decomposition.
4. Quantify the potential financial impact of alternative decisions.
5. Generate actionable recommendations based on **configurable business objectives, rules, and constraints**.
6. Continuously update its findings as new transactions arrive.
7. Support future integration with an interactive dashboard where management can configure recommendation preferences.

Do not hardcode any specific branch, product, existing anomaly, or recommendation.

**Core principle:** Stages 1–4 discover and quantify business problems. Stage 5 determines which actions to recommend according to management's configurable priorities.

---

## 2. Dataset and Preprocessing

Input file: `stories_transactions.csv`

Columns:

- `Transaction ID`
- `Date`
- `Time`
- `Branch`
- `Customer ID`
- `Order Type`
- `Product`
- `Category`
- `Quantity`
- `Unit Price (USD)`
- `Discount Rate`
- `Unit Cost (USD)`
- `Line Revenue (USD)`
- `Line Gross Profit (USD)`

Each row represents a product line. Multiple rows may belong to the same transaction.

### Preprocessing requirements

- Validate column types, quantities, timestamps, prices, and costs.
- Preserve legitimate repeated transaction IDs.
- Detect duplicate transaction lines where distinguishable.
- Verify revenue and gross-profit arithmetic, allowing for monetary rounding.
- Aggregate data by branch, product, category, and reporting period.
- Calculate quantity-weighted unit prices, costs, and realized discounts when appropriate.
- Calculate revenue, cost of goods sold, gross profit, and gross margin.
- Handle missing products, new branches, and incomplete reporting periods.
- Keep raw observations separate from aggregated analytical features.

The existing dataset contains approximately 51,177 product lines, three branches, and 45 products.

**Important:** Unit cost and listed selling price are constant within each branch-product combination in the current dataset. Do not invent temporal cost or price variation.

---

## 3. Stage 1 — Feature Construction

For each branch-product-period combination, compute:

- Total units sold.
- Total revenue.
- Total cost of goods sold.
- Total gross profit.
- Gross profit margin.
- Listed unit price.
- Unit cost.
- Realized revenue per unit.
- Effective discount rate.
- Share of branch sales volume.
- Share of branch revenue.
- Share of branch gross profit.

Create matched-product comparison tables to ensure equivalent products are compared across branches.

Support configurable reporting periods, with monthly summaries as the default.

---

## 4. Stage 2 — Statistical Models

Implement robust two-way fixed-effects regression.

### A. Unit-cost model

\[
\log C_{b,p}=\alpha_p+\beta_b+\epsilon_{b,p}
\]

Where:

- \(C_{b,p}\): Unit cost for product \(p\) at branch \(b\).
- \(\alpha_p\): Product-specific effect.
- \(\beta_b\): Branch-specific cost effect.
- \(\epsilon_{b,p}\): Residual.

Fit using Huber regression with an identifiable fixed-effects parameterization.

Train on distinct branch-product cost observations, rather than counting repeated identical costs as independent evidence.

The model should identify:

- Systematically elevated branch costs.
- Product-specific cost anomalies.
- Branch-wide cost multipliers.
- Categories contributing to cost differences.

### B. Selling-price model

Implement the equivalent model:

\[
\log P_{b,p}=\gamma_p+\delta_b+\eta_{b,p}
\]

Identify systematic selling-price differences and exceptional product-level deviations.

Keep discounts separate from listed prices.

### C. Model comparisons

Compare robust fixed-effects regression with ordinary least squares.

Report:

- Estimated branch effects.
- Product effects.
- Residuals.
- Fit statistics.
- Stability under different benchmark choices.

Avoid treating the limited number of independent branch-product combinations as thousands of independent cost observations.

---

## 5. Stage 3 — Profitability Decomposition

For each branch, calculate:

\[
GP_b=\sum_p Q_{b,p}(P_{b,p}-C_{b,p})-D_b
\]

Where \(D_b\) represents total discount amounts.

Use actual recorded revenue and gross profit as authoritative after validating their accounting consistency.

Explain differences between branches in terms of:

1. Unit costs.
2. Listed selling prices.
3. Discounts.
4. Product mix.
5. Sales volume, when comparing total profit.

Use matched products and standardized baskets to ensure meaningful comparisons.

Implement a consistent decomposition methodology, such as Shapley-style or exact sequential decomposition.

The component contributions must reconcile with the total profitability difference and avoid double-counting.

Provide category-level and product-level breakdowns.

---

## 6. Stage 4 — Counterfactual Scenario Analysis

Evaluate possible changes such as:

- Matching another branch's unit costs.
- Matching another branch's selling prices.
- Matching median peer costs.
- Reducing selected category costs.
- Adjusting product selling prices.
- Adjusting discount policies.
- Changing product mix under specified assumptions.

Calculate:

- Current gross profit.
- Counterfactual gross profit.
- Absolute profit difference.
- Percentage profit difference.
- Product and category contributions.
- Important assumptions.

For accounting scenarios, keep quantities fixed unless an explicit demand-response assumption is supplied.

For price changes, calculate break-even demand changes rather than assuming unchanged demand.

Clearly separate:

- Observed financial differences.
- Accounting counterfactuals.
- Hypothetical intervention benefits.

Do not claim that a proposed price change will causally increase profit without supporting evidence.

---

# 7. Stage 5 — Configurable Recommendation Engine

**This stage must be designed for future integration with a management interface.**

The recommendation logic, thresholds, objectives, and priorities must NOT be hardcoded into the core implementation.

Instead, Stage 5 must accept an independent configuration object that determines how recommendations are generated, filtered, and ranked.

The engine must work with default settings immediately, but every business-facing rule should be modifiable through configuration.

## 7.1 Default Business Objective

Default objective:

**Maximize expected gross-profit improvement while prioritizing recommendations supported by strong evidence and reasonable implementation feasibility.**

The engine must support alternative objectives, including:

- `maximize_gross_profit`
- `reduce_costs`
- `increase_revenue`
- `improve_gross_margin`
- `conservative`

Changing the objective must alter recommendation selection and ranking without requiring model retraining.

## 7.2 Default Configuration

Create an external file named `recommendation_config.yaml`.

Use the following initial settings:

```yaml
business_objective: maximize_gross_profit

scope:
  branches: all
  categories: all
  products: all

reporting:
  period: monthly
  currency: USD

detection:
  minimum_units_sold: 30
  minimum_financial_exposure_usd: 100

  cost_difference_threshold_pct: 10
  price_difference_threshold_pct: 5
  gross_margin_gap_threshold_pp: 5
  discount_difference_threshold_pp: 5

  minimum_matched_products: 5

recommendation_weights:
  financial_impact: 0.50
  evidence_strength: 0.30
  actionability: 0.20

risk:
  tolerance: moderate
  allow_price_experiments: true
  allow_discount_experiments: true
  allow_cost_reduction_recommendations: true
  allow_product_mix_experiments: true
  allow_automatic_price_changes: false

scenario_defaults:
  cost_reduction_percentages: [2, 5, 10]
  price_change_percentages: [-5, -3, 3, 5]
  maximum_suggested_price_change_pct: 5

ranking:
  maximum_recommendations: 10
  prioritize_unresolved_findings: true
  suppress_duplicate_alerts: true
  impact_normalization: log
  minimum_evidence_score: 0.60

recommendation_status:
  enabled: true
  available_statuses:
    - new
    - under_review
    - testing
    - resolved
    - dismissed
```

These are initial operational defaults, not statistically validated optimal thresholds.

The implementation must validate configuration values and handle invalid or missing parameters gracefully.

## 7.3 Recommendation Generation

Use outputs from Stages 2–4 to generate candidate recommendations.

Potential recommendation types:

### A. Elevated unit costs

Condition: Matched-product costs exceed the configured benchmark by a meaningful amount.

Possible actions:

- Investigate supplier contracts.
- Examine recipe and ingredient costs.
- Verify branch-specific accounting rules.
- Prioritize products with the greatest financial exposure.

### B. Unusually low selling prices

Condition: Product prices are materially lower than comparable branches.

Possible actions:

- Investigate pricing differences.
- Evaluate alternative prices.
- Recommend controlled price experiments.

Always calculate break-even demand changes.

### C. Excessive discount exposure

Condition: Discounts create unusually large gross-margin differences.

Possible actions:

- Review discount policies.
- Identify affected categories.
- Recommend promotion-effectiveness experiments.

### D. Unfavorable product mix

Condition: A branch's profitability differs materially because of its product composition.

Possible actions:

- Identify high-margin categories with low sales shares.
- Recommend merchandising experiments.
- Investigate whether product mix reflects legitimate branch-specific demand.

### E. Systematic profitability discrepancy

Condition: A branch has a persistent profitability disadvantage across matched products.

Possible actions:

- Determine the dominant sources of the discrepancy.
- Combine related product-level findings.
- Recommend targeted management investigation.

Generate candidate actions only when supported by the observed evidence.

## 7.4 Financial Evaluation of Recommendations

Every candidate recommendation must include a quantified financial measure where possible.

Examples:

**Unit-cost difference:**

\[
\Delta GP=\sum_p Q_p(C_p-C_{\text{benchmark},p})
\]

**Price-change scenario:**

\[
GP_{\text{new}}=Q_{\text{assumed,new}}(P_{\text{new}}-C)
\]

The latter must explicitly state its demand assumption.

If the intervention effect cannot be identified from the dataset, report an accounting scenario or break-even condition instead of a fabricated expected gain.

## 7.5 Recommendation Ranking

Use a configurable ranking function.

Default:

\[
\boxed{S_i=0.50I_i+0.30E_i+0.20A_i}
\]

Where:

- \(I_i\): Normalized financial importance.
- \(E_i\): Evidence-strength score.
- \(A_i\): Actionability score.

All three components must be normalized to the range \([0,1]\).

Financial importance should use a stable, documented transformation, such as logarithmic normalization against a configurable reference scale.

Evidence strength should consider:

- Number of comparable products.
- Consistency of discrepancies.
- Data completeness.
- Stability across reporting periods.
- Sensitivity to benchmark choice.

Actionability should consider:

- Whether the underlying factor is controllable.
- Required management effort.
- Implementation risk.
- Whether experimentation is necessary.

Do not present the evidence score as a statistically calibrated probability unless calibration has actually been performed.

The weights must be configurable.

Validate that they are nonnegative and sum to 1.

Different business objectives should have documented default weighting profiles, which users can override.

**Important:** Recommendations with unknown causal benefits must not automatically outrank verified financial discrepancies simply because their hypothetical profit potential is large.

## 7.6 Example Recommendation Output

Return structured recommendations such as:

```json
{
  "recommendation_id": "REC-001",
  "branch": "Garden",
  "type": "high_unit_cost",
  "scope": "branch",
  "finding": "Systematically elevated unit costs",
  "evidence": {
    "matched_products": 45,
    "estimated_cost_difference_pct": 43
  },
  "financial_impact": {
    "amount_usd": 15923,
    "period": "six_months",
    "type": "fixed_volume_counterfactual"
  },
  "recommended_action": "Investigate branch-level costing, supplier contracts, and recipes.",
  "assumptions": [
    "Peer unit costs are used as the benchmark",
    "Sales quantities remain unchanged"
  ],
  "evidence_score": 0.9,
  "actionability_score": 0.8,
  "priority_score": 0.87,
  "status": "new"
}
```

The values above illustrate the expected output structure. Scores must be computed by the implementation rather than copied from this example.

## 7.7 Interface-Ready Configuration

Design Stage 5 around a clean Python interface:

```python
recommendations = engine.generate(
    findings=findings,
    scenarios=scenarios,
    config=user_config
)
```

Provide programmatic functions such as:

```python
get_recommendation_config()

validate_recommendation_config(config)

update_recommendation_config(config)

generate_recommendations(
    findings,
    scenarios,
    config
)

rank_recommendations(
    recommendations,
    config
)
```

A future frontend must be able to modify:

- Business objective.
- Target branches.
- Detection thresholds.
- Financial-impact thresholds.
- Ranking weights.
- Risk tolerance.
- Allowed recommendation types.
- Scenario assumptions.
- Maximum displayed recommendations.

The current project does not require a frontend.

However, **the recommendation engine must be fully usable by a future API or web interface without changes to its analytical core**.

Use JSON-compatible configuration structures and provide clear validation errors.

Keep configuration persistence separate from business logic.

## 7.8 Recommendation Lifecycle

New transactions should not cause previously detected unchanged discrepancies to be reported as new findings repeatedly.

Implement persistent recommendation identities and statuses.

A recommendation can be:

- New.
- Under review.
- Testing.
- Resolved.
- Dismissed.

Track changes in evidence, financial exposure, and priority over time.

Keep issue state separate from generated analytical outputs, so that regenerating a report does not erase management decisions.

Do not automatically reopen dismissed or resolved recommendations unless materially new evidence appears.

---

## 8. Stage 6 — Interactive Reporting

Generate a Plotly-based HTML report.

The report must contain:

### Company overview

- Revenue by branch.
- Gross profit by branch.
- Gross margin comparisons.
- Ranked discrepancies.
- Executive recommendations.

### Cost analysis

- Matched-product cost comparisons.
- Learned branch-cost effects.
- Product-level cost heatmaps.
- Category-level excess-cost contributions.
- Profit opportunity Pareto charts.

### Pricing analysis

- Branch price comparisons.
- Price-versus-cost plots.
- Learned branch-price effects.
- Break-even demand sensitivity.

### Profitability decomposition

- Cost contributions.
- Price contributions.
- Discount contributions.
- Product-mix contributions.
- Waterfall charts.

### Recommendation dashboard

- Ranked recommendations.
- Supporting evidence.
- Financial exposure.
- Recommended action.
- Assumptions and limitations.
- Recommendation status.

Include the configuration used to generate the report.

Optionally create a simple configuration demonstration showing how changing the business objective changes recommendation rankings.

Do not implement a full frontend unless it is required for demonstrating the configuration system.

---

## 9. Continuous Data Updates

Implement a reusable data-update pipeline.

When new transactions arrive:

1. Validate incoming records.
2. Update analytical summaries.
3. Handle newly appearing products and branches.
4. Refresh regression models when underlying cost or price information changes.
5. Recalculate financial comparisons.
6. Recalculate counterfactual scenarios.
7. Regenerate recommendations using the active configuration.
8. Update recommendation status and history.
9. Regenerate reports.

Do not retrain unnecessarily when only sales quantities change and the relevant fitted structural relationships remain unchanged.

Support rerunning Stage 5 with different management preferences **without rerunning Stages 1–4**.

---

## 10. Testing and Evaluation

Implement tests covering:

- Accounting consistency.
- Fixed-effects model correctness.
- Matched-product comparisons.
- Profit decomposition reconciliation.
- Counterfactual calculations.
- New-data ingestion.
- Configuration validation.
- Recommendation generation.
- Recommendation ranking.
- Recommendation deduplication.
- Recommendation lifecycle persistence.

### Configurability test

Run the recommendation system using at least three configurations:

1. Maximize gross profit.
2. Reduce costs.
3. Conservative / low-risk.

Verify that the resulting recommendation rankings reflect the chosen objective while the underlying measured financial findings remain unchanged.

Also verify that modifying thresholds and ranking weights does not require model retraining.

---

## 11. Implementation Requirements

Use Python.

Suggested dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `statsmodels`
- `scipy`
- `plotly`
- `pyyaml`
- `pydantic`
- `pytest`

Suggested project structure:

```text
stories-intelligence/
├── data/
│   └── stories_transactions.csv
├── config/
│   └── recommendation_config.yaml
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── fixed_effects.py
│   ├── profit_decomposition.py
│   ├── scenario_analysis.py
│   ├── recommendation_engine.py
│   ├── recommendation_config.py
│   ├── recommendation_storage.py
│   └── visualization.py
├── tests/
├── outputs/
├── main.py
├── requirements.txt
└── README.md
```

Maintain clear separation between:

- Statistical learning.
- Deterministic financial analysis.
- Business decision policies.
- Configuration management.
- Presentation.

---

## 12. Deliverables

Produce:

1. Fully executable Python project.
2. Trained fixed-effects models.
3. Detailed profitability decomposition.
4. Counterfactual scenario engine.
5. Configurable recommendation engine.
6. Default YAML configuration.
7. Example alternative CEO configurations.
8. Machine-readable recommendations in JSON.
9. Interactive Plotly HTML report.
10. Model and financial summary CSV files.
11. Automated tests.
12. Documentation explaining how to integrate Stage 5 with a future frontend.

---

## 13. Acceptance Criteria

The project is successful if:

- It automatically discovers cross-branch anomalies without hardcoded findings.
- The model learns and quantifies systematic cost and price differences.
- Profitability differences are explained mathematically.
- Recommendations are backed by evidence and financial scenarios.
- Default recommendation rules work immediately.
- Business objectives, thresholds, and priorities can be changed externally.
- Stage 5 can regenerate recommendations without retraining Stages 1–4.
- Existing recommendations are tracked rather than repeatedly duplicated.
- The system adapts when new transactions arrive.
- Outputs are interpretable and suitable for CEO-level decision-making.

**Final design requirement:** Build Stages 1–4 as an objective analytical engine, and Stage 5 as an independent, configurable business decision layer. The CEO must eventually be able to change recommendation preferences through an interface without altering the underlying statistical models.
