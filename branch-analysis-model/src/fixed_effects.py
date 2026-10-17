"""Two-way product/branch fixed effects with OLS and dependency-light Huber IRLS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FixedEffectsResult:
    target: str
    method: str
    reference_branch: str
    reference_product: str
    branch_effects: dict[str, float]
    branch_multipliers: dict[str, float]
    product_effects: dict[str, float]
    intercept: float
    rmse: float
    mae: float
    r_squared: float
    iterations: int
    observations: int
    residuals: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _design(panel: pd.DataFrame, reference_branch: str, reference_product: str):
    branches = sorted(panel["Branch"].unique())
    products = sorted(panel["Product"].unique())
    bcols = [b for b in branches if b != reference_branch]
    pcols = [p for p in products if p != reference_product]
    columns = [np.ones(len(panel))]
    columns.extend((panel["Branch"].to_numpy() == b).astype(float) for b in bcols)
    columns.extend((panel["Product"].to_numpy() == p).astype(float) for p in pcols)
    return np.column_stack(columns), branches, products, bcols, pcols


def fit_fixed_effects(
    panel: pd.DataFrame,
    target: str,
    method: str = "huber",
    reference_branch: str | None = None,
    reference_product: str | None = None,
    huber_delta: float = 1.345,
    max_iter: int = 200,
    tolerance: float = 1e-9,
) -> FixedEffectsResult:
    clean = panel[["Branch", "Product", target]].drop_duplicates().dropna().copy()
    clean = clean.loc[clean[target] > 0].sort_values(["Product", "Branch"]).reset_index(drop=True)
    if clean.empty:
        raise ValueError(f"No positive observations available for {target}")
    branches, products = sorted(clean["Branch"].unique()), sorted(clean["Product"].unique())
    reference_branch = reference_branch or branches[0]
    reference_product = reference_product or products[0]
    if reference_branch not in branches or reference_product not in products:
        raise ValueError("Reference branch/product must occur in the model panel")
    x, branches, products, bcols, pcols = _design(clean, reference_branch, reference_product)
    y = np.log(clean[target].to_numpy(float))
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    iterations = 1
    if method == "huber":
        for iterations in range(1, max_iter + 1):
            residual = y - x @ coef
            median = np.median(residual)
            scale = 1.4826 * np.median(np.abs(residual - median))
            if scale < 1e-12:
                break
            standardized = np.abs(residual) / scale
            weights = np.ones_like(standardized)
            mask = standardized > huber_delta
            weights[mask] = huber_delta / standardized[mask]
            root_w = np.sqrt(weights)
            updated = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)[0]
            if np.max(np.abs(updated - coef)) < tolerance:
                coef = updated
                break
            coef = updated
    elif method != "ols":
        raise ValueError("method must be 'huber' or 'ols'")
    fitted = x @ coef
    residual = y - fitted
    ss_total = float(np.sum((y - y.mean()) ** 2))
    b_effect = {reference_branch: 0.0}
    b_effect.update({b: float(coef[1 + i]) for i, b in enumerate(bcols)})
    p_start = 1 + len(bcols)
    p_effect = {reference_product: 0.0}
    p_effect.update({p: float(coef[p_start + i]) for i, p in enumerate(pcols)})
    residual_rows = []
    for row, fit, resid in zip(clean.itertuples(index=False), fitted, residual):
        residual_rows.append({
            "branch": row.Branch, "product": row.Product, "observed": float(getattr(row, target)),
            "fitted": float(np.exp(fit)), "log_residual": float(resid),
            "multiplicative_residual": float(np.exp(resid) - 1),
        })
    return FixedEffectsResult(
        target=target, method=method, reference_branch=reference_branch,
        reference_product=reference_product, branch_effects=b_effect,
        branch_multipliers={k: float(np.exp(v)) for k, v in b_effect.items()},
        product_effects=p_effect, intercept=float(coef[0]),
        rmse=float(np.sqrt(np.mean(residual ** 2))), mae=float(np.mean(np.abs(residual))),
        r_squared=float(1 - np.sum(residual ** 2) / ss_total) if ss_total else 1.0,
        iterations=iterations, observations=len(clean), residuals=residual_rows,
    )


def compare_models(panel: pd.DataFrame, target: str) -> dict[str, Any]:
    robust = fit_fixed_effects(panel, target, "huber")
    ols = fit_fixed_effects(panel, target, "ols")
    branches = sorted(panel["Branch"].unique())
    pairwise_runs = []
    for reference in branches:
        fit = fit_fixed_effects(panel, target, "huber", reference_branch=reference)
        pairwise_runs.append({
            f"{a}__vs__{b}": fit.branch_effects[a] - fit.branch_effects[b]
            for i, a in enumerate(branches) for b in branches[i + 1:]
        })
    stability = {
        pair: float(np.ptp([run[pair] for run in pairwise_runs]))
        for pair in pairwise_runs[0]
    } if pairwise_runs else {}
    return {"huber": robust.to_dict(), "ols": ols.to_dict(), "benchmark_stability_log_effect_range": stability}


def predict_fixed_effects(model: FixedEffectsResult | dict[str, Any], panel: pd.DataFrame) -> pd.DataFrame:
    """Score distinct branch-product observations without fitting on the scored split."""
    model_dict = model.to_dict() if isinstance(model, FixedEffectsResult) else model
    target = model_dict["target"]
    clean = panel[["Branch", "Product", target]].drop_duplicates().dropna().copy()
    clean = clean.loc[clean[target] > 0].sort_values(["Product", "Branch"]).reset_index(drop=True)
    branch_effects = model_dict["branch_effects"]
    product_effects = model_dict["product_effects"]
    clean["known_branch"] = clean["Branch"].isin(branch_effects)
    clean["known_product"] = clean["Product"].isin(product_effects)
    clean["scorable"] = clean["known_branch"] & clean["known_product"]
    clean["observed"] = clean[target].astype(float)
    clean["observed_log"] = np.log(clean["observed"])
    clean["predicted_log"] = np.nan
    mask = clean["scorable"]
    clean.loc[mask, "predicted_log"] = (
        float(model_dict["intercept"])
        + clean.loc[mask, "Branch"].map(branch_effects).astype(float)
        + clean.loc[mask, "Product"].map(product_effects).astype(float)
    )
    clean["predicted"] = np.exp(clean["predicted_log"])
    clean["log_residual"] = clean["observed_log"] - clean["predicted_log"]
    clean["absolute_error"] = (clean["observed"] - clean["predicted"]).abs()
    clean["absolute_percentage_error"] = clean["absolute_error"] / clean["observed"]
    return clean[[
        "Branch", "Product", "observed", "predicted", "observed_log", "predicted_log",
        "log_residual", "absolute_error", "absolute_percentage_error", "known_branch",
        "known_product", "scorable",
    ]]


def evaluate_fixed_effects(model: FixedEffectsResult | dict[str, Any], panel: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions = predict_fixed_effects(model, panel)
    scored = predictions[predictions["scorable"]].copy()
    if scored.empty:
        metrics = {
            "observations": len(predictions), "scored_observations": 0,
            "coverage": 0.0, "log_rmse": None, "log_mae": None,
            "rmse_original_units": None, "mae_original_units": None,
            "mape_pct": None, "r_squared_log": None,
        }
        return metrics, predictions
    log_residual = scored["log_residual"].to_numpy(float)
    residual = (scored["observed"] - scored["predicted"]).to_numpy(float)
    actual_log = scored["observed_log"].to_numpy(float)
    ss_total = float(np.sum((actual_log - actual_log.mean()) ** 2))
    metrics = {
        "observations": len(predictions), "scored_observations": len(scored),
        "coverage": float(len(scored) / len(predictions)),
        "log_rmse": float(np.sqrt(np.mean(log_residual ** 2))),
        "log_mae": float(np.mean(np.abs(log_residual))),
        "rmse_original_units": float(np.sqrt(np.mean(residual ** 2))),
        "mae_original_units": float(np.mean(np.abs(residual))),
        "mape_pct": float(100 * scored["absolute_percentage_error"].mean()),
        "r_squared_log": float(1 - np.sum(log_residual ** 2) / ss_total) if ss_total else 1.0,
        "unseen_branches": sorted(predictions.loc[~predictions["known_branch"], "Branch"].unique().tolist()),
        "unseen_products": sorted(predictions.loc[~predictions["known_product"], "Product"].unique().tolist()),
    }
    return metrics, predictions
