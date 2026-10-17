# Decision engine capability and split audit

Audit date: 2026-10-17. This is a read-only repository audit; it did not fit,
train, tune, or execute a policy.

## Existing split

The files in `decision-model/data/` are the existing transaction-level
chronological 70/15/15 split. Their immutable counts, boundaries, and SHA-256
fingerprints are recorded in `config/split_manifest.json`. Calendar dates overlap
at the two boundaries because transactions are the split unit, but the checked
timestamps remain ordered: TRAIN ends 2026-05-16 11:18:42, VALIDATION starts
2026-05-16 18:02:38; VALIDATION ends 2026-06-09 07:56:07, and TEST starts
2026-06-09 22:30:42.

## Available specialist components

- `aya/` forecasts business-as-usual daily branch-product units and reports
  uncertainty/anomalies. It is not action-conditioned, it predicts sales rather
  than latent demand, and it uses a whole-day split that differs from the shared
  transaction split. An adapter must be explicitly chosen before use.
- `branch-analysis-model/` exports branch-product listed-price and unit-cost
  context and residuals. Its own contract explicitly contains no decisions or
  action response. It cannot establish price elasticity. The currently generated
  `outputs/decision_model_input.json` identifies its data window as TEST, so it
  must not be consumed by PPO training; a TRAIN-only export would be needed for
  training-time state features.
- `mohammad/` provides basket association evidence. Its historical split-safe
  workflow may provide signals, but its current full-data descriptive workflow
  includes former validation/test observations and must not be used to fit PPO
  training features.
- `PricingOptimization/` currently contains only split CSVs and a manifest; no
  callable pricing or demand-response simulator was found.

## Blocking capability result

No existing component forecasts action-conditioned product quantities. A price
or discount edit could therefore not be assigned a credible demand response,
and no bundle, promotion, cross-sell, assortment, inventory, staffing, or
procurement response model was found. Real-effect optimization is intentionally
unconfigured. `RealSimulatorAdapter` accepts a future callable only after its
declared capabilities are supplied and can be audited with
`audit_action_response`.

`SyntheticSimulator` is available only for unit tests and demonstrations. Every
forecast it emits is labeled `synthetic_only_unvalidated_intervention_effects`.
Its coefficients are explicit assumptions and are not calibrated from the
transaction files.

## Input wiring status

The user has not yet selected source files for the state inputs. The
`BusinessStateAssembler` therefore consumes typed in-memory records and opens no
files. The configuration keeps all input paths null and training disabled.
