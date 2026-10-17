"""Technical Plotly report for predictor diagnostics only; no decision policy."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _chart(div_id: str, traces: list[dict], layout: dict) -> str:
    return (
        f'<div id="{div_id}" class="chart"></div>'
        f'<script>Plotly.newPlot({json.dumps(div_id)}, {json.dumps(traces)}, {json.dumps(layout)}, '
        '{responsive:true, displaylogo:false});</script>'
    )


def generate_predictor_report(
    predictor: pd.DataFrame, models: dict, evaluation: dict, output_path: str | Path,
) -> Path:
    metric_rows = []
    for split_key, split_label in (("validation_metrics", "Validation"), ("test_metrics", "Test")):
        for model_name in ("cost", "price"):
            for method in ("huber", "ols"):
                metric = evaluation[split_key][model_name][method]
                metric_rows.append(
                    f"<tr><td>{split_label}</td><td>{model_name.title()}</td><td>{method.upper()}</td>"
                    f"<td>{metric['scored_observations']}/{metric['observations']}</td>"
                    f"<td>{metric['log_rmse']:.6f}</td><td>{metric['mae_original_units']:.6f}</td>"
                    f"<td>{metric['mape_pct']:.4f}%</td><td>{metric['r_squared_log']:.6f}</td></tr>"
                )
    cost_effects = models["cost"]["huber"]["branch_multipliers"]
    price_effects = models["price"]["huber"]["branch_multipliers"]
    charts = [
        _chart("effects", [
            {"type": "bar", "x": list(cost_effects), "y": [100 * (v - 1) for v in cost_effects.values()], "name": "Cost effect"},
            {"type": "bar", "x": list(price_effects), "y": [100 * (v - 1) for v in price_effects.values()], "name": "Price effect"},
        ], {"title": "Training-fitted branch multipliers", "barmode": "group", "yaxis": {"title": "% vs reference"}}),
        _chart("cost_prediction", [{
            "type": "scatter", "mode": "markers", "x": predictor["predicted_unit_cost"].round(5).tolist(),
            "y": predictor["unit_cost"].round(5).tolist(), "text": (predictor["Branch"] + " / " + predictor["Product"]).tolist(),
        }], {"title": "Test: predicted versus observed unit cost", "xaxis": {"title": "Predicted"}, "yaxis": {"title": "Observed"}}),
        _chart("price_prediction", [{
            "type": "scatter", "mode": "markers", "x": predictor["predicted_listed_unit_price"].round(5).tolist(),
            "y": predictor["listed_unit_price"].round(5).tolist(), "text": (predictor["Branch"] + " / " + predictor["Product"]).tolist(),
        }], {"title": "Test: predicted versus observed listed price", "xaxis": {"title": "Predicted"}, "yaxis": {"title": "Observed"}}),
        _chart("residuals", [
            {"type": "box", "y": predictor["cost_residual_pct"].round(5).tolist(), "name": "Cost residual"},
            {"type": "box", "y": predictor["price_residual_pct"].round(5).tolist(), "name": "Price residual"},
        ], {"title": "Test residual distributions", "yaxis": {"title": "%"}}),
    ]
    disclosure = html.escape(evaluation["evaluation_design"]["important_disclosure"])
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stories Predictor Diagnostics</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{margin:0;background:#f5f2ec;color:#28241f;font-family:Inter,Segoe UI,sans-serif}}main{{max-width:1150px;margin:auto;padding:32px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}.card{{background:white;border-radius:14px;padding:18px;box-shadow:0 3px 14px #0001;margin:18px 0}}.chart{{height:410px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#43382d;color:white}}</style></head>
<body><main><h1>Stories Predictor Diagnostics</h1><p>This artifact contains model outputs only. It applies no thresholds, rankings, recommendations, or management decisions.</p>
<section class="card"><h2>Chronological holdout evaluation</h2><p>Models fit on train only; validation used for diagnostics; test used for final scoring.</p><p><strong>Disclosure:</strong> {disclosure}</p>
<table><thead><tr><th>Split</th><th>Target</th><th>Method</th><th>Coverage</th><th>Log RMSE</th><th>MAE</th><th>MAPE</th><th>Log R²</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></section>
<section class="grid">{''.join(f'<div class="card">{chart}</div>' for chart in charts)}</section></main></body></html>"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
