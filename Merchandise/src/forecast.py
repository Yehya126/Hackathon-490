"""Next-day demand models: baselines, LightGBM mean model and a negative binomial demand distribution."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from features import FEATURES, OFFSET_FEATURES

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


class DemandModel:
    """LightGBM with a Poisson objective.

    mode="raw":      predicts units directly from the features.
    mode="offset":   predicts a multiplier on the baseline (log(baseline) is the starting score),
                     so growth and the weekday pattern come from the baseline and the trees only
                     learn adjustments; forecasts can exceed anything seen in training.
    mode="linear":   raw mode with linear trees, which can extend trends.
    """

    def __init__(self, mode: str = "offset"):
        self.mode = mode
        self.features = OFFSET_FEATURES if mode == "offset" else FEATURES
        self.params = dict(LGBM_PARAMS, linear_tree=True, linear_lambda=1.0) if mode == "linear" else LGBM_PARAMS
        self.booster, self.rounds = None, None

    def _dataset(self, df, reference=None):
        init = np.log(df["baseline"]) if self.mode == "offset" else None
        return lgb.Dataset(df[self.features], df["units"], init_score=init, reference=reference)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, rounds: int | None = None):
        """Fit on rows with full history. With `valid`, early-stop to pick the round count."""
        train = train.dropna(subset=["lag_28"])
        dtrain = self._dataset(train)
        if valid is not None:
            self.booster = lgb.train(self.params, dtrain, num_boost_round=3000,
                                     valid_sets=[self._dataset(valid, dtrain)],
                                     callbacks=[lgb.early_stopping(100, verbose=False)])
            self.rounds = self.booster.best_iteration
        else:
            self.rounds = rounds
            self.booster = lgb.train(self.params, dtrain, num_boost_round=rounds)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(df[self.features], num_iteration=self.rounds, raw_score=True)
        if self.mode == "offset":
            raw = raw + np.log(df["baseline"].to_numpy())
        return np.exp(raw)


def fit_lgbm(train: pd.DataFrame, valid: pd.DataFrame | None = None, rounds: int | None = None,
             mode: str = "offset"):
    """Convenience wrapper: returns (model, rounds)."""
    model = DemandModel(mode).fit(train, valid, rounds)
    return model, model.rounds


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
