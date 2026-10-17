# Registry-driven PPO decision engine

This directory contains the untrained registry-driven decision engine.
Importing any module is side-effect free: it
does not load a data split, fit a scaler/model, run an environment, or start PPO.

## Current status

- The existing split is fingerprinted in the shared `../data/split_manifest.json`; see
  `CAPABILITY_AUDIT.md` for the model and leakage audit.
- `config/decisions.json` is the editable top-level registry array. Unknown
  metadata is preserved, so compatible registry additions do not require PPO
  core edits. It is the single canonical decision registry and includes evidence
  requirements and decision kinds. Required fields and integer
  parameters are validated strictly.
- Integer proposals come from runtime simulator/operational support intervals.
  There is no fixed value grid. The finite candidate count only limits the
  number presented in one PPO step.
- The environment selects up to ten new actions. STOP is normally masked until
  three actions have been selected, then becomes available; if no feasible
  candidate remains earlier, an emergency STOP prevents an invalid all-false
  action mask.
  It evaluates the joint portfolio against the same-seed no-new-action forecast,
  gives one weekly reward, and carries multi-week actions forward.
- Product forecasts reconcile units, effective price, revenue, COGS, and gross
  profit and can be exported at SKU/category/branch/company levels.
- Real action optimization is not wired because no existing model predicts
  action-conditioned demand. The synthetic simulator is test-only and labels all
  effects as unvalidated assumptions.
- Training is disabled in `config/ppo_decision.json`. The three input paths and
  the reward scale are intentionally null until their sources are specified.

## Input integration contract

Construct `SpecialistProductInput` records from the chosen files/services and
pass them to `BusinessStateAssembler`. Supply runtime categorical targets and an
`IntegerDomain` for every supported numeric parameter. A missing credible domain
masks that action instead of extrapolating or clamping it.

Registry parameters with `value_source` are never sampled by the decision model.
Their integer values must arrive in `ProposalContext.sourced_parameter_values`
with the registry-required evidence. This is how the current price-change entry
can require the future store price/performance model's recommendation.

Wrap the production action-response function in `RealSimulatorAdapter`. It must
return one `ForecastRow` for every branch-product pair. Before exposing a
capability to the candidate generator, run `audit_action_response`; price actions
must change both effective price and units.

## Tests

From the repository root, without installing the PPO packages:

```powershell
python -m unittest discover -s decision-model/tests -v
```

To eventually train after inputs, a TRAIN-derived reward scale, supported
integer domains, and a validated action-conditioned simulator are configured:

```powershell
python -m pip install -r decision-model/requirements.txt
```

Then call `train_maskable_ppo` explicitly with `TrainingRequest(enabled=True,
...)`. No CLI auto-trains, and no TEST path is accepted for fitting or rollout.
