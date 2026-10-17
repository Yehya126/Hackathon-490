"""Multi-day forecasting: expected units 1-14 days ahead, using only data known on the forecast day.

For a target day t forecast h days ahead, the forecast is made at the end of the origin day t - h.
Every sales-based input is taken "as of" the origin day; calendar inputs (weekday, day of month,
trend, weekday index) belong to the target day, since they are known in advance.
One LightGBM model covers all horizons, with `horizon` as an input.
"""

from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import CATEGORICAL, GROWTH_CENTER_DAYS
from forecast import LGBM_PARAMS
from panel import KEYS

MAX_HORIZON = 14

# Inputs measured at the origin day (each column at row t uses data up to t-1, so shifting by
# h-1 rows gives the value built from data up to t-h, the origin day).
ORIGIN_COLUMNS = ["lag_1", "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_7",
                  "branch_roll_mean_7", "branch_roll_mean_28", "branch_growth", "price_ratio",
                  "unit_margin_prev", "lag_1_rel"]
TARGET_COLUMNS = ["weekday", "day_of_month", "trend", "wd_index"]
HORIZON_FEATURES = (CATEGORICAL + TARGET_COLUMNS + ORIGIN_COLUMNS
                    + ["horizon", "same_weekday_last", "same_weekday_mean_4", "same_weekday_rel",
                       "roll_mean_7_rel", "baseline"])


def horizon_frame(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Rows of `df` (features from add_features) re-expressed as h-day-ahead forecasts."""
    df = df.sort_values(KEYS + ["Date"])
    g = df.groupby(KEYS, sort=False, observed=True)
    out = df[KEYS + ["Category", "shelf_class", "Date", "units"] + TARGET_COLUMNS].copy()
    for col in ORIGIN_COLUMNS:
        out[col] = g[col].shift(h - 1)
    # Same weekday as the target: the most recent four such days known at the origin.
    k = math.ceil(h / 7)
    same = [g["units"].shift(7 * (k + i)) for i in range(4)]
    out["same_weekday_last"] = same[0]
    out["same_weekday_mean_4"] = pd.concat(same, axis=1).mean(axis=1, skipna=False)
    level = out["roll_mean_28"].clip(lower=0.1)
    growth_days = GROWTH_CENTER_DAYS + h - 1
    out["baseline"] = level * out["wd_index"] * np.exp(out["branch_growth"].fillna(0) * growth_days)
    out["same_weekday_rel"] = out["same_weekday_mean_4"] / out["baseline"]
    out["roll_mean_7_rel"] = out["roll_mean_7"] / out["baseline"]
    out["horizon"] = h
    out["origin"] = out["Date"] - pd.Timedelta(days=h)
    return out


def stacked(df: pd.DataFrame, horizons=range(1, MAX_HORIZON + 1)) -> pd.DataFrame:
    return pd.concat([horizon_frame(df, h) for h in horizons], ignore_index=True)


class HorizonModel:
    """Baseline-offset LightGBM (Poisson) over stacked horizons."""

    def __init__(self):
        self.booster, self.rounds = None, None

    @staticmethod
    def _dataset(frame, reference=None):
        return lgb.Dataset(frame[HORIZON_FEATURES], frame["units"], init_score=np.log(frame["baseline"]),
                           reference=reference)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, rounds: int | None = None):
        train = train.dropna(subset=["roll_mean_28", "units"])
        dtrain = self._dataset(train)
        if valid is not None:
            valid = valid.dropna(subset=["roll_mean_28", "units"])
            self.booster = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=3000,
                                     valid_sets=[self._dataset(valid, dtrain)],
                                     callbacks=[lgb.early_stopping(100, verbose=False)])
            self.rounds = self.booster.best_iteration
        else:
            self.rounds = rounds
            self.booster = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=rounds)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(frame[HORIZON_FEATURES], num_iteration=self.rounds, raw_score=True)
        return np.exp(raw + np.log(frame["baseline"].to_numpy()))


def future_frame(df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Forecast rows for origin+1 ... origin+14, each at its own horizon."""
    parts = []
    for h in range(1, MAX_HORIZON + 1):
        f = horizon_frame(df, h)
        parts.append(f[f["Date"] == origin + pd.Timedelta(days=h)])
    return pd.concat(parts, ignore_index=True)
