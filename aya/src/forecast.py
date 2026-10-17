"""Next-day demand models: baselines, LightGBM mean model and a negative binomial demand distribution."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from features import FEATURES

LGBM_PARAMS = dict(
    objective="poisson", learning_rate=0.03, num_leaves=15, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, seed=7,
)


def seasonal_naive(df: pd.DataFrame) -> pd.Series:
    """Same weekday last week."""
    return df["lag_7"]


def weekday_average(df: pd.DataFrame) -> pd.Series:
    """Mean of the last four same weekdays, scaled by the recent 28-day trend."""
    trend = (df["roll_mean_7"] / df["roll_mean_28"]).clip(0.5, 2).fillna(1)
    return df["same_weekday_mean_4"] * trend


def fit_lgbm(train: pd.DataFrame, valid: pd.DataFrame | None = None, rounds: int | None = None):
    """Fit on rows with full history. With `valid`, early-stop and return the best round count."""
    train = train.dropna(subset=["lag_28"])
    dtrain = lgb.Dataset(train[FEATURES], train["units"])
    if valid is not None:
        dvalid = lgb.Dataset(valid[FEATURES], valid["units"], reference=dtrain)
        model = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        return model, model.best_iteration
    return lgb.train(LGBM_PARAMS, dtrain, num_boost_round=rounds), rounds


def fit_dispersion(y: pd.Series, mu: pd.Series, groups: pd.Series) -> pd.Series:
    """Method-of-moments NB dispersion per group: Var = mu + alpha * mu^2."""
    frame = pd.DataFrame({"excess": (y - mu) ** 2 - mu, "mu2": mu ** 2, "g": groups.astype(str)})
    sums = frame.groupby("g")[["excess", "mu2"]].sum()
    return (sums["excess"] / sums["mu2"]).clip(lower=1e-3)


def _nb(mu, alpha):
    n = 1.0 / np.asarray(alpha, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    return stats.nbinom(n, n / (n + mu))


def nb_quantile(mu, alpha, q) -> np.ndarray:
    return _nb(mu, alpha).ppf(q)


def nb_cdf(k, mu, alpha) -> np.ndarray:
    return _nb(mu, alpha).cdf(k)


def nb_sf(k, mu, alpha) -> np.ndarray:
    """P(Y >= k)."""
    return _nb(mu, alpha).sf(np.asarray(k) - 1)


def poisson_vs_nb_loglik(y, mu, alpha) -> tuple[float, float]:
    """Mean log-likelihood under Poisson and NB, to justify the distribution choice."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    return float(stats.poisson(mu).logpmf(y).mean()), float(_nb(mu, alpha).logpmf(y).mean())


def accuracy(df: pd.DataFrame, pred_col: str, by: str = "Branch") -> pd.DataFrame:
    err = df[pred_col] - df["units"]
    out = pd.DataFrame({"abs": err.abs(), "err": err, "units": df["units"], by: df[by]})
    g = out.groupby(by, observed=True).sum()
    table = pd.DataFrame({"MAE": out.groupby(by, observed=True)["abs"].mean(),
                          "WAPE": g["abs"] / g["units"], "bias": g["err"] / g["units"]})
    total = pd.DataFrame({"MAE": [out["abs"].mean()], "WAPE": [out["abs"].sum() / out["units"].sum()],
                          "bias": [out["err"].sum() / out["units"].sum()]}, index=["ALL"])
    return pd.concat([table, total]).round(3)
