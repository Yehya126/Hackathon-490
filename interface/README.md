# Stories Studio

The Decisions page shows LLM-generated recommendations: Codex reviewed the
outputs of Marketing Expert, Merchandise, Pricing Optimization and Branch
Analysis and proposed actions with source references, assumptions and pilot
steps. These are saved in `manual_decisions.js`; they are not live LLM responses
or PPO output. Importing Yehya's actual decisions replaces this temporary view.
Do not treat scenario profit gains as measured outcomes.

**Website entry point: `interface/index.html`.**

From the repository root, run `python start_website.py`, then open
http://127.0.0.1:8000/. Keep that terminal running. Stop with Ctrl+C.
Use `--port 8001` if needed. The launcher serves this folder only and needs
no packages beyond Python. See the root README for the file map.

Workflow additions: Home has a four-step navigation guide. Pricing shows a
scenario comparison chart; Branch analysis shows the largest cost differences.
Forecast, pricing and branch tables support search, sorting, ten-row pagination,
reset and export of the visible rows. Pricing also filters raise/cut/keep.
Charts summarize the selected branch; table search filters only the table.
These presentation controls do not change model scores or execute decisions.

Latest sync: `Merchandise` replaces `aya`. Pricing reads the 135 published rows
in `PricingOptimization/outputs/price_recommendations.csv`; its gains are
scenario estimates based on assumed price sensitivity, not measured uplift.
Branch comparisons now read `cleaned_predictor_results.csv`. The decision engine
is implemented but untrained, with no generated real decisions to show.
Regenerate `data.js` with `python interface/build_data.py` after pulling changes.

The build reads Marketing_Expert, Merchandise, PricingOptimization, the branch
predictor and Yehya's decision registry. `synced.js` adapts their contracts.
Registry entries are action definitions, not generated decisions. The manual
import schema below remains available for Yehya's decisions; committed forecast
and branch outputs take precedence in their views.

Local interface. Open `index.html`, or run `python -m http.server 8000 --directory interface`.
Refresh Mohammad's real data with `python interface/build_data.py`. No training runs.

The visual design uses warm neutral tones, short copy and generous spacing.
Mohammad's pairings, category patterns and dine-in results are connected.
Other models have honest empty states until their outputs are supplied.

Use **Connect model output** to import JSON. Imported outputs stay in this
browser's local storage; they are not shared with teammates or sent to a server.
Each import replaces the previous output for that model. No decisions are
automatically applied to prices or business operations.

## Integration contract

All records need `branch`: `Overall`, `Garden`, `Harbour` or `Uptown`.
Amounts are USD; dates use YYYY-MM-DD. Examples below describe structure only;
do not treat example values as model results.

Aya: `{ "model": "aya", "forecasts": [{ "branch": "Garden", "date": "2026-07-01", "product": "Latte", "units": 0, "revenue_usd": 0 }], "pricing": [{ "branch": "Garden", "product": "Latte", "current_price": 0, "suggested_price": 0, "reason": "Model explanation" }] }`

YY: `{ "model": "yy", "signals": [{ "branch": "Garden", "title": "Signal title", "summary": "Model explanation", "severity": "Review", "period": "Observation period" }] }`

Yehya: `{ "model": "yehya", "decisions": [{ "branch": "Garden", "title": "Proposed action", "reason": "Decision rationale", "sources": ["Mohammad", "Aya", "YY"], "status": "Proposed", "next_step": "Pilot action" }] }`

Final decisions are rendered only from Yehya's output. The interface does not
substitute its own decisions or invent forecasts. Current Mohammad results are
full-data descriptive analysis, not independent validation. Forecast intervals,
time horizons and value estimates should be supplied by model authors before
using them to support business claims.
