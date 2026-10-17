# Stories Studio

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
