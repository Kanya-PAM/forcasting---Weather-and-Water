"""Reservoir model: data-driven inflow + fitted release rule + mass balance.

Inflow  (log space, fitted by ridge regression)
    log1p(I_t) = b0 + b1 log1p(I_{t-1}) + b2 log1p(P_t) + b3 log1p(P_{t-1})
                 + b4 log1p(P_{t-2}) + b5 log1p(API_t) + seasonal harmonics + e_t
    P = catchment rain (from the weather ensemble), API = antecedent
    precipitation index. e_t is AR(1) noise so ensemble spread is realistic.

Release (fitted by OLS)
    R_t = c0 + c1 R_{t-1} + c2 (S_{t-1} - Target_t) + harmonics + u_t

Storage (mass balance, MCM/day)
    S_t = S_{t-1} + I_t - R_t - E_t,  with spill above 102 % and floor at dead storage
    E_t = evaporation_mm(month) * surface_area(S) * 1e-3
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from . import config as C
from .data import antecedent_index, catchment_rain, evaporation, rule_target
from .weather_model import harmonics


class DamModel:
    def __init__(self, dam: C.Dam):
        self.dam = dam

    def _inflow_X(self, lag_i, p0, p1, p2, api, H):
        return np.column_stack([np.log1p(lag_i), np.log1p(p0), np.log1p(p1), np.log1p(p2),
                                np.log1p(api), H])

    def fit(self, dam_df: pd.DataFrame, rain: pd.Series) -> "DamModel":
        df = dam_df.join(rain.rename("rain"), how="inner").dropna()
        idx = df.index
        p = catchment_rain(df["rain"].to_numpy())
        api = antecedent_index(p)
        H = harmonics(idx, 2)
        I = df["inflow"].to_numpy()
        X = self._inflow_X(I[1:-1], p[2:], p[1:-1], p[:-2], api[2:], H[2:])
        y = np.log1p(I[2:])
        self.inflow = Ridge(alpha=1.0).fit(X, y)
        res = y - self.inflow.predict(X)
        self.res_phi = np.corrcoef(res[1:], res[:-1])[0, 1]
        self.res_sd = res.std() * np.sqrt(1 - self.res_phi ** 2)
        self.inflow_r2 = self.inflow.score(X, y)

        S, R = df["storage"].to_numpy(), df["release"].to_numpy()
        tgt = rule_target(self.dam, idx)
        Xr = np.column_stack([R[:-1], S[:-1] - tgt[1:], H[1:]])
        self.release = LinearRegression().fit(Xr, R[1:])
        self.rel_sd = (R[1:] - self.release.predict(Xr)).std()
        self.release_r2 = self.release.score(Xr, R[1:])
        self.rmin = np.percentile(R, 1)
        self.last_res = res[-1]
        return self

    def simulate(self, hist: pd.DataFrame, hist_rain: pd.Series, rain_ens: np.ndarray,
                 dates: pd.DatetimeIndex, rng: np.random.Generator) -> dict:
        n, T = rain_ens.shape
        d = self.dam
        hr = hist_rain.loc[:hist.index[-1]].to_numpy()
        # catchment rain for forecast days needs 2 days of history for the 3-day mean
        full = np.column_stack([np.tile(hr[-2:], (n, 1)), rain_ens])
        pc = 0.4 * full + 0.6 * (full + np.roll(full, 1, 1) + np.roll(full, 2, 1)) / 3
        pc = pc[:, 2:]
        p_hist = catchment_rain(hr)
        api = np.full(n, antecedent_index(p_hist)[-1])
        H = harmonics(dates, 2)
        tgt = rule_target(d, dates)

        S = np.full(n, hist["storage"].iloc[-1])
        I = np.full(n, hist["inflow"].iloc[-1])
        R = np.full(n, hist["release"].iloc[-1])
        p1 = np.full(n, p_hist[-1])
        p2 = np.full(n, p_hist[-2])
        e = np.full(n, self.last_res)
        out = {k: np.zeros((n, T)) for k in ("storage", "inflow", "release", "evap")}
        for j in range(T):
            api = 0.93 * api + pc[:, j]
            X = self._inflow_X(I, pc[:, j], p1, p2, api, np.tile(H[j], (n, 1)))
            e = self.res_phi * e + rng.normal(0, self.res_sd, n)
            I = np.expm1(self.inflow.predict(X) + e).clip(0)
            Xr = np.column_stack([R, S - tgt[j], np.tile(H[j], (n, 1))])
            R = (self.release.predict(Xr) + rng.normal(0, self.rel_sd, n)).clip(self.rmin)
            E = evaporation(d, S, pd.DatetimeIndex([dates[j]] * n))
            S_new = S + I - R - E
            spill = np.clip(S_new - 1.02 * d.capacity_mcm, 0, None)
            R, S_new = R + spill, S_new - spill
            S = np.maximum(S_new, d.dead_mcm)
            p2, p1 = p1, pc[:, j]
            for k, v in (("storage", S), ("inflow", I), ("release", R), ("evap", E)):
                out[k][:, j] = v
        return out


def summarize(dam: C.Dam, ens: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    s = ens["storage"]
    cap = dam.capacity_mcm
    usable = cap - dam.dead_mcm
    return pd.DataFrame({
        "date": dates,
        "storage_mean_mcm": s.mean(0), "storage_p10_mcm": np.percentile(s, 10, 0),
        "storage_p90_mcm": np.percentile(s, 90, 0),
        "storage_pct_capacity": s.mean(0) / cap * 100,
        "usable_mcm": s.mean(0) - dam.dead_mcm,
        "usable_pct": (s.mean(0) - dam.dead_mcm) / usable * 100,
        "inflow_mean_mcm": ens["inflow"].mean(0), "inflow_p10_mcm": np.percentile(ens["inflow"], 10, 0),
        "inflow_p90_mcm": np.percentile(ens["inflow"], 90, 0),
        "release_mean_mcm": ens["release"].mean(0), "evap_mean_mcm": ens["evap"].mean(0),
        "prob_above_100pct": (s > cap).mean(0) * 100,
        "prob_above_80pct": (s > 0.8 * cap).mean(0) * 100,
    })
