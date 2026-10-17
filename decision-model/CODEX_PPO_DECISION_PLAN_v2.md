# Codex Implementation Plan v2 — Stories Coffee PPO Decision Engine

> **Scope of this revision:** Maintain an **editable array of decisions**, allow **integer-valued action parameters without hardcoded percentage/value grids**, use the existing **train / validation / test split without leakage**, and evaluate each weekly decision portfolio using **next-week product-level projected quantities, prices, revenue, costs and gross profit**, compared to the **same-week no-decision forecast**.

## 0. Mission and non-negotiable design choices

Implement a single hierarchical PPO decision engine (company / branch / category / product) that chooses zero to three coordinated business actions each week. It must read numerical inputs from the existing cross-branch, basket, merchandising, pricing and forecasting components and propose actions to improve *forecast next-week business outcomes*. The agent **proposes** decisions; it must not apply changes to the live business.

**Non-negotiable:** Historic Stories prices and unit costs are fixed within each branch-product in the supplied January–June transactions. A model that forecasts a price or units with no proposed action is **not** automatically able to predict how demand would respond to a price change, bundle, promotion, or supplier switch. Do not present simulator assumptions as learned causal effects or proven profit gains. If the simulator is not action-conditioned, implement a separately marked synthetic demo environment and report this limitation prominently.

The current inputs are already split into **TRAIN, VALIDATE, TEST**. **Never train or fit any model, scaler, simulator response rule or policy on TEST.** Use TRAIN for fitting and training; VALIDATE for model and hyperparameter selection; TEST exactly once for final evaluation after freezing choices. Do not merge datasets or generate lag features that look into future outcomes.

## 1. Repository audit and split discipline

Before implementing:

1. Discover the existing train, validate and test files/indices, check their dates and schemas, and document their sizes. **Do not recreate or reshuffle the split.** If it is not chronological, flag the issue rather than silently changing it.
2. Identify the available specialized models and the next-week forecasting/simulator interface, their model checkpoints and which split each was fit on.
3. Check whether the simulator forecasts only next-week price, or predicts **action-conditioned sales quantity** as well. Identify supported action families and state variables.
4. Verify that price/cost/discount forecasts are defined consistently. The agent's proposed price adjustment must actually change the price used in the sales forecast; the baseline price forecast must not overwrite a chosen action.
5. Record all unavailable intervention effects; don't fabricate a demand response or numeric accuracy figures.

Train-only fitting applies to imputation, encoding, normalization, any ML price or quantity predictor, response estimator, reward scaling, PPO training episodes, and synthetic scenario calibration. VALIDATE may drive tuning or early stopping, but is not training data. TEST cannot guide feature engineering, action selection rules, training-time simulator parameters, reward shaping, thresholds or hyperparameters. Save immutable split manifests and test-access logs.

## 2. Editable decision registry: a single ARRAY, not hardcoded branches

Put **every decision definition** in `config/decisions.json`, as one easily edited top-level JSON array. Also expose `load_decisions(path)` with strict schema validation. The PPO code, simulator adapters, masks, report generator and dashboard integration must read the registry; **no if/else action catalogue embedded in policy code**. A family may still have a typed executor registered by `type`.

Create a ready-to-edit file matching the provided `decision_registry.json` template (included with this plan). Registry entries declare:

- `id`, `label`, `enabled`, `scopes`, `parameters`, `simulator_capability`, `interaction_group`, `executor`.
- Every **numeric decision parameter** has `dtype: "int"`. IDs, target names and action-type strings remain categorical strings.
- Percentages are **integer percentage points** (e.g. `7` means +7%, `-12` means −12%). Durations are integer weeks; quantities integer units; currency amounts integer cents when exposed as editable parameters.
- **Do not hardcode a finite set of choices** such as `[-5,-3,0,3,5]` or arbitrary caps such as `max=10` or `max=20`. Integers must be configurable as integers, including values the model hasn't tried yet.
- Distinguish **parameter domain validity** from arbitrary preset value grids: discounts must result in an actual rate in [0%, 100%]; prices must remain strictly positive; durations/units must be positive where relevant; supplier options must exist; modeled response must be within validated support. Optional CEO policies (price ceilings, budgets, maximum acceptable change) are separate editables and may restrict actions when enabled.
- If no credible finite simulator-supported range exists for a parameter, do **not** allow unlimited extrapolation. Mark it unsupported/out-of-support and request a bounded experiment or assumption; do not silently clamp or pretend any integer is valid.
- A registry edit must be visible in generated reports and action JSON. Adding a new family needs an executor and simulator capability but must not require editing PPO core loops. Expanding the registry or action semantics generally requires PPO retraining and reevaluation.

Initial configurable decision families:

| Action | Typical integer parameters | Scopes | Simulator requirement |
|---|---|---|---|
| `price_change` | `change_pct`, `duration_weeks` | product/category/branch/company | action-conditioned demand and prices |
| `discount_change` | `change_percentage_points`, `duration_weeks` | product/category/branch | action-conditioned demand and effective revenue |
| `bundle_create` | `bundle_discount_pct`, `duration_weeks` | branch/category | incremental bundle uptake + substitution |
| `bundle_modify` | `bundle_discount_pct`, `duration_weeks` | branch | bundle-response and interaction model |
| `cross_sell` | `duration_weeks`, `intensity_pct` | product/branch | add-on acceptance + cannibalization |
| `promotion` | `intensity_pct`, `duration_weeks` | product/category/branch | promotional demand response and execution cost |
| `assortment_change` | `duration_weeks` | branch/product | substitution and product availability |
| `preparation_change` | `units_delta`, `duration_weeks` | branch/product | inventory, waste, stockouts |
| `staffing_change` | `staff_delta`, `duration_weeks` | branch | service capacity and payroll |
| `procurement_scenario` | `target_cost_reduction_pct`, `duration_weeks` | branch/category/product/company | validated feasible supplier/operational change; else scenario only |
| `no_new_action` / `stop` | none | company | always permitted |

Targets (branch/category/product IDs) and lists of products in bundles are **categorical identifiers**, not numeric parameters. `procurement_scenario` must never be treated as a guaranteed real cost reduction based on peer benchmarking alone. Staff/inventory/assortment decisions only enter RL training when the simulator can meaningfully process their effects.

## 3. How PPO handles unconstrained *integer choices* technically

A standard finite `Discrete` PPO policy **cannot choose an unbounded integer directly**. Implement integer freedom **without a manually fixed value grid** as follows:

**Version 1 — dynamic proposal candidates with MaskablePPO:**

1. Generate eligible candidate *templates* from enabled decision definitions and valid targets.
2. At each decision step, propose **integer parameter values dynamically** from the simulator-supported and operationally feasible integer domain, with exploration across the domain and refinement around promising values. Use deterministic seeds; include sign diversity and large/small values where feasible. NEVER restrict to a static literal list of percentages.
3. Construct finite `(template, target, integer parameters)` candidate records for the current PPO decision step. The finite candidate-pool size is a **computational limit on simultaneous proposals, not a bound on integer parameter values**; make it configurable. Log the explored ranges and coverage.
4. Sort deterministically, pad the observation, include candidate integer-value features and mask infeasible candidates. PPO chooses a candidate index or STOP.
5. Correctly label this as **optimization over sampled candidate integers**, not guaranteed optimization over all integers. Evaluate whether widening candidate coverage improves performance.

**Optional later version — custom parameterized PPO:** Build a conditional integer-valued parameter head (with appropriate discrete action likelihoods, masks and stable updates) if dynamic candidate sampling misses useful parameter choices. Do not round a Gaussian policy and pretend its unrounded log-probabilities are the correct probabilities of the executed integer actions. The training and evaluation pipeline must be compatible with either version.

Do not insert a fixed percentage grid in examples, config or tests as a compulsory action set. Tests may specify a small toy finite feasible domain **only to establish a known optimum**.

## 4. State representation

Weekly state contains only information available **at decision time**:

- Company and branch summaries: revenue, gross profit, margin, order counts, historical trends.
- Branch-product-level recent quantities, realized price, listed price, unit cost, discount, category, share of branch revenue, and sales mix.
- Specialized-model numerical outputs: branch cost/price discrepancies, basket association signals, merchandising signals, and any supported price-response or demand forecasts, plus missingness/uncertainty flags.
- Known next-week external context (calendar, confirmed plans); not next week's observed sales.
- Currently active actions, remaining durations, outstanding budgets, approved implementation constraints.
- Current candidate list: type/scope/target/parameters, projected financial features if justified, supporting evidence, conflicts and mask.

Normalize using TRAIN statistics only. Do not supply actual VALIDATE/TEST future outcomes as features when simulating decisions.

## 5. Next-week predictions: required outcome contract

The central decision evaluation is **next week's product-by-product projected sales**, under each complete decision portfolio, and a paired no-new-decision baseline.

Required simulator interface:

```python
simulate_next_week(
    current_state: BusinessState,
    actions: list[CandidateAction],
    exogenous_scenario,
    seed: int,
) -> NextWeekForecast
```

`NextWeekForecast` must return a row for each **branch × product (SKU)**, with at least:

```python
ForecastRow(
    branch_id: str,
    product_id: str,
    predicted_units: float,              # expected number of units, >= 0
    predicted_list_price_usd: float,     # or action-adjusted known menu price
    predicted_effective_price_usd: float,# after discounts/bundle allocations
    predicted_unit_cost_usd: float,
    predicted_revenue_usd: float,
    predicted_cogs_usd: float,
    predicted_gross_profit_usd: float,
    uncertainty: dict,
)
```

Implement consistent aggregations by category/branch/company:

\[
\widehat{Revenue}_{t+1}(A)=\sum_{b,p}\hat Q_{b,p,t+1}(A)\,\hat P^{effective}_{b,p,t+1}(A)
\]

\[
\widehat{GP}_{t+1}(A)=\widehat{Revenue}_{t+1}(A)-\sum_{b,p}\hat Q_{b,p,t+1}(A)\,\hat C_{b,p,t+1}(A)
\]

For bundles, allocate bundle revenue consistently or use explicitly modeled bundle line items without double counting; account for cross-sell, substitution, discounts and stockout effects. Use real dollar amounts (round only for presentation). Historical gross profit may be used to validate arithmetic but not to train an ML model to relearn its exact definition.

**Important:** If the existing forecast model predicts only market/menu prices, that is insufficient for evaluating interventions. Actions must flow into the **demand prediction** \(\hat Q_{b,p,t+1}(A)\); otherwise different candidate decisions would produce bogus projected rewards. Add an adapter only if action-conditioned behavior is supported; otherwise use a clearly labeled synthetic environment and do not present its expected revenue/profit as validated.

## 6. Compare decisions against next week's no-decision forecast

For a given week and current state, use the **same** exogenous scenario/random shocks in both simulation calls:

```python
baseline = simulate_next_week(state, [], scenario, seed)
policy   = simulate_next_week(state, selected_portfolio, scenario, seed)

expected_revenue_change = policy.total_revenue - baseline.total_revenue
expected_profit_change = policy.total_gross_profit - baseline.total_gross_profit
net_incremental_profit = expected_profit_change - action_implementation_costs
```

The primary KPI the user explicitly wants is the **expected revenue difference**, and the dashboard must display it clearly, with both per-product and branch/company breakdowns. **However, revenue alone is not a sufficient reward:** e.g., a very large discount can increase revenue while destroying gross profit. Default PPO reward should therefore continue to use **incremental net gross profit**, with expected revenue recorded as a secondary metric and configurable alternative business objective.

Define weekly reward with a paired full-episode no-intervention trajectory (rather than simply last week's profit):

\[
r_t=\frac{[\widehat{GP}^{policy}_{t+1}-K^{policy}_{t+1}]-[\widehat{GP}^{baseline}_{t+1}-K^{baseline}_{t+1}]}{S}-\lambda Risk_t
\]

`S` is fixed and fit/calibrated from TRAIN only; `lambda` and the CEO objective are configurable. No artificial exponential reward. `Risk_t` must be calculable, never a fictional learned probability. Ensure expenses already included in gross profit are not subtracted again.

When constructing three actions in one week, evaluate the complete **joint portfolio**, not the sum of isolated action effects. Support expected revenue/profit and lower/upper scenario estimates when simulators provide uncertainty.

## 7. Weekly hierarchical decision process

Implement a Gymnasium environment with two time scales:

1. PPO observes the current state and the dynamic integer-valued candidate list.
2. It selects the first candidate or STOP.
3. Refresh masks and candidate features; PPO selects a second, a third or STOP. Maximum **three new actions per simulated week** is the initial configuration, not a compulsory number of actions.
4. Advance the calendar **only after** STOP or the maximum number of selections.
5. Apply new and previously active actions together; reject overlaps/contradictions unless an explicit precedence/interaction rule is implemented.
6. Run the action-conditioned next-week forecast and paired baseline; calculate forecast **units, prices, revenue and profit** per product and in aggregate.
7. Give PPO one weekly joint reward and move to the next simulated state. Repeat for a configurable multi-week horizon, initially eight weeks.

Use `gymnasium`, `sb3-contrib` `MaskablePPO`, PyTorch and a padded candidate-list observation; expose `action_masks()` for invalid proposal entries. Both training and inference must respect the current registry and operational/simulator feasibility checks. Use a true no-op option even with unused action slots.

Preserve active interventions across weeks; a policy run might continue while a second week adds a complementary action. Simulated next state must change after a decision, or PPO will not be able to learn long-term effects.

## 8. Data split and training/evaluation (strict)

| Stage | Permitted data | Prohibited |
|---|---|---|
| Preprocessing/feature fitting | TRAIN | Fitting on VALIDATE/TEST |
| Forecast/response simulator fitting | TRAIN | TEST outcomes, TEST-derived effects |
| PPO training episodes / reward scaling | TRAIN-based initial states and fitted/synthetic TRAIN scenarios | VALIDATE or TEST episode training |
| Hyperparameter/threshold/model selection | VALIDATE | TEST |
| Final evaluation (single frozen run) | TEST | Updating any weights, policies, normalizers or settings afterward |

If test data are actual historical transactions without interventions, TEST can evaluate next-week forecast accuracy on recorded business-as-usual conditions. It **cannot** validate hypothetical price/bundle causal effects. For PPO intervention evaluation, build disjoint **training, validation and testing simulator scenario sets**, with held-out simulator parameters/seeds, and report **simulated-only** performance separately from historic forecast tests. Do not treat an unseen random seed alone as proof of generalization to unknown economics.

Evaluate actual one-week forecasts for: per-product units (MAE, WAPE where defined), price where forecasted (MAE), total revenue/GP error; report by branch and category. Prevent lookahead at transaction timestamps, price adjustment dates, and roll-forward lag features. Train on TRAIN; tune on VALIDATE; generate predictions and assess on TEST **without updating**.

At training time optimize RL parameters only through simulator calls derived from TRAIN assumptions, with validation-based early stopping/tuning. Save split fingerprints, forecast model hashes, simulator version, PPO checkpoint and registry version.

## 9. Baselines to evaluate decisions

On identical held-out simulated states and matched exogenous conditions compare:

1. No action.
2. Random feasible decisions with integer-valued parameters.
3. Best single action under joint simulator.
4. Greedy up-to-three (adding actions sequentially and re-simulating the combined set).
5. Exhaustive combination and integer-parameter search **only on deliberately small toy finite domains**, to establish an exact optimum for tests; never claim a global optimum across all possible integers.
6. PPO policy.

Compare expected revenue change, net incremental profit, downside risk, action count, policy interactions, and runtime. Evaluate multiple random seeds/scenario assumptions including unseen response profiles, not merely the same simulator trajectory used for training. Report honest confidence intervals over independent episodes.

## 10. Output: product-level forecast + portfolio comparison

Provide structured JSON and CSV showing both the **baseline** and **action** forecast for the next week for every product and branch, plus portfolio totals. Example schema (not example financial results):

```json
{
  "week_start": "2026-07-06",
  "decisions": [
    {
      "type": "price_change",
      "scope": "product",
      "target": {"branch": "Garden", "product": "Iced Latte"},
      "parameters": {"change_pct": 7, "duration_weeks": 1}
    }
  ],
  "product_forecasts": [
    {
      "branch": "Garden",
      "product": "Iced Latte",
      "baseline": {"units": null, "effective_price_usd": null, "revenue_usd": null, "gross_profit_usd": null},
      "with_actions": {"units": null, "effective_price_usd": null, "revenue_usd": null, "gross_profit_usd": null}
    }
  ],
  "company_totals": {
    "baseline_predicted_revenue_usd": null,
    "policy_predicted_revenue_usd": null,
    "delta_expected_revenue_usd": null,
    "baseline_predicted_gross_profit_usd": null,
    "policy_predicted_gross_profit_usd": null,
    "delta_expected_gross_profit_usd": null,
    "net_incremental_profit_usd": null,
    "reward": null
  },
  "simulator_support": "action_conditioned_or_synthetic_only",
  "assumptions": [],
  "validation_status": "unverified_intervention_effects"
}
```

Do not make up numerical forecasts; set null only where a simulator genuinely cannot provide them and explain why. Provide individual SKU, category and branch reports; show changes in quantity, price, revenue, COGS, GP and net return. Include saved decision registry path/hash, policy checkpoint, data split ID, training/validation/test provenance and uncertainty limitations.

## 11. Configuration

Create `config/ppo_decision.yaml` (or JSON) with options independent from the editable `config/decisions.json` array:

```yaml
frequency: weekly
planning_horizon_weeks: 8
max_new_actions_per_week: 3
candidate_proposals_per_step: 64   # computational pool size, NOT a parameter bound
integer_parameter_proposals:
  method: dynamic_feasible_sampling
  static_value_grids: forbidden
  exploration: stratified_plus_local_refinement
  only_simulator_supported_values: true
discount_factor_gamma: 0.99
reward_objective: incremental_net_gross_profit
track_expected_revenue_as_primary_report_metric: true
reward_scale: training_derived_fixed
risk_aversion_lambda: 0.0
registry_file: config/decisions.json
constraints:
  positive_final_price: true
  valid_discount_rate: true
  no_conflicting_actions: true
  action_budget_usd: null  # may be set by CEO; do not invent a cap
  require_human_approval: true
training:
  split_manifest: data/splits/manifest.json
  seed: 42
  smoke_test_timesteps: 1000
  full_training_timesteps: 100000
  tuning_split: validate
  test_split: test
  forbid_test_in_training: true
simulator:
  mode: audit_and_use_if_action_conditioned_else_synthetic_demo
  report_synthetic_assumptions: true
```

This config is illustrative. Adapt file names to the existing repository rather than creating fake splits or overwriting available split files. All actionable settings should be usable through an API/web interface in the future. Changing registry definitions, action semantics or candidate generation can require PPO retraining; changing report-only preferences does not.

## 12. Code organization

```text
config/
  decisions.json                  # Editable top-level ARRAY
  ppo_decision.yaml               # General PPO / company constraints
src/decision_rl/
  schemas.py
  split_guard.py                  # Enforces train/validate/test access
  registry.py                     # Loads and validates editable decision array
  candidate_proposals.py          # Dynamic integer-valued candidate generator
  validity.py                     # Constraints, simulator support, conflict masks
  specialist_adapters.py
  simulator_contract.py
  real_simulator_adapter.py
  synthetic_simulator.py
  observation.py
  environment.py
  reward.py
  train.py
  validate.py
  test_once.py
  baseline_optimizers.py
  reporting.py
outputs/
  forecasts_by_product_week.csv
  portfolio_comparisons.json
  checkpoints/
  validation_reports/
  frozen_test_report/
tests/
  test_registry_changes.py
  test_integer_parameters.py
  test_strict_split_isolation.py
  test_forecast_revenue_profit.py
  test_action_conditioning.py
  test_action_interactions.py
  test_reward_and_baseline.py
  test_mask_and_stop.py
  test_small_exact_optimum.py
README_decision_rl.md
```

## 13. Required acceptance tests

1. **Editable registry:** Adding/removing/enabling an action in `decisions.json` changes available candidates automatically. Invalid registry entries produce readable validation errors.
2. **No preset integer grid:** The system can propose/use an integer parameter not contained in any previous hardcoded list (e.g., +7%); all numeric action parameters are true integers, not floats. It rejects nonsensical results such as negative prices or discounts outside 0–100%.
3. **Dynamic candidate validity:** PPO sees an action only when simulator support and real operational constraints are satisfied. The finite candidate-pool limit is not imposed as a limit on allowed integer magnitudes.
4. **Frozen TEST:** Tests deliberately place a recognizable sentinel in TEST; no training scaler, forecast model, simulator fitter, reward normalizer, PPO rollout or tuning process may read it. TEST is used only in final frozen evaluation.
5. **Weekly SKU forecasts:** For every product, correct predicted units × action-adjusted effective price = predicted revenue (or explicitly documented bundle allocation), and revenue minus COGS = GP. Category/branch/company totals reconcile.
6. **Paired comparison:** With no action, baseline and policy forecasts match and incremental reward is zero (up to numerical tolerance). Identical exogenous scenarios are used for each comparison.
7. **Action effectiveness:** For a toy simulator with known negative price elasticity, a price change updates both effective price **and units**, and produces the mathematically expected revenue/profit change. If a simulator only changes prices and not demand, it must fail capability checks for decision optimization.
8. **Multiple decisions:** Combined portfolios are simulated jointly; conflicting actions cannot silently stack; STOP works; agent may choose 0–3 new actions; persistent effects carry into later weeks.
9. **Reward correctness:** Positive net incremental gross profit receives positive reward; negative receives negative reward. Expected revenue deltas are always shown, even if they disagree in sign with profit deltas.
10. **Validation/test integrity:** Hyperparameters chosen via VALIDATE only; final TEST result generated once, not used for selection or retraining.
11. **Honest labels:** Synthetic forecasts are labeled synthetic; projected improvements are not reported as realized financial gains.
12. **Baselines:** PPO is compared with no action, random, best single, greedy and toy exhaustive search. No assumption that PPO must win.

## 14. Codex execution sequence and handoff

**M1** — Inspect repo and locate existing TRAIN/VALIDATE/TEST splits and simulator; provide capability/data leakage report.

**M2** — Create `config/decisions.json` **top-level decision array**, schema, loader, validity checks and integer parameter proposal generator; verify arbitrary valid integer values work without fixed grids.

**M3** — Implement action-conditioned **next-week branch-product forecasts** with predicted units, effective price, revenue, COGS and GP, alongside a paired same-week no-action forecast.

**M4** — Implement `Gymnasium` weekly environment, portfolio builder (max 3 actions, STOP), joint reward and persistent state. Run tests and show product-level financial reconciliation.

**M5** — Train PPO on TRAIN-based environments only; tune on VALIDATE; freeze setup before TEST. First learn on a tiny synthetic scenario with known optimum.

**M6** — Evaluate policy against no-action/random/greedy/single/exhaustive-small baselines, using held-out simulated behavior and a separate final TEST forecast audit.

**M7** — Export model checkpoints, registry, next-week forecast tables, paired action-vs-baseline comparisons, validation/test reports, audit logs, and a readable README; show exact commands.

**Implement M1–M4 first and demonstrate the calculations. Do not proceed to full PPO training if the existing simulator cannot respond to action choices; use a labeled synthetic sandbox instead. Preserve the existing train/validate/test split and never train on TEST.**

**Registry convention:** All action families are listed as `enabled: true` in the supplied editable registry. This **does not** override simulator capability checks: families without an action-response model are excluded from real-effect optimization and may only be used in clearly labeled synthetic tests. To remove a family, set `enabled: false` or remove its registry object.
