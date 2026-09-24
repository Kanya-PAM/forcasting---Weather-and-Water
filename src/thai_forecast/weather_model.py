"""Stochastic weather model (weather generator conditioned on the latest state).

Rainfall
  * occurrence: logistic regression  P(wet_t) = f(seasonal harmonics, wet_{t-1},
    fraction of wet days in last 7 days)
  * amount on wet days: Gamma distribution fitted (MLE) to wet days within
    +/-20 days of the calendar day
Temperature (Tmax, Tmin, separately)
  * climatology: OLS on 3 seasonal harmonics + linear trend
  * anomaly: AR(2) + effect of same-day rain anomaly (wet - seasonal P(wet)), Gaussian noise with a
    month-specific standard deviation

Forecasts are Monte-Carlo ensembles: every member draws rain occurrence,
amount and temperature noise day by day, starting from the observed state on
the day before the forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

WET_MM = 1.0


def harmonics(index: pd.DatetimeIndex, k: int = 3) -> np.ndarray:
    w = 2 * np.pi * index.dayofyear.to_numpy() / 365.25
    return np.column_stack([f(j * w) for j in range(1, k + 1) for f in (np.sin, np.cos)])


class WeatherModel:
    def fit(self, df: pd.DataFrame) -> "WeatherModel":
        df = df.dropna()
        idx = df.index
        self.t0 = idx[0]
        wet = (df["rain"].to_numpy() >= WET_MM).astype(float)
        H = harmonics(idx)

        # --- rain occurrence
        prev = np.r_[0, wet[:-1]]
        frac7 = pd.Series(prev).rolling(7, min_periods=1).mean().to_numpy()
        Xo = np.column_stack([H, prev, frac7, prev * H[:, 0], prev * H[:, 1]])
        self.occ = LogisticRegression(C=10.0, max_iter=2000).fit(Xo[7:], wet[7:])
        # seasonal wet-day probability, so temperature sees the rain *anomaly*
        self.pseas = LogisticRegression(max_iter=2000).fit(H, wet)
        wet_anom = wet - self.pseas.predict_proba(H)[:, 1]

        # --- rain amounts: keep wet-day values and their day-of-year
        self.wet_doy = idx.dayofyear.to_numpy()[wet > 0]
        self.wet_amt = df["rain"].to_numpy()[wet > 0]
        self._gamma_cache = {}

        # --- temperature
        yrs = ((idx - self.t0).days.to_numpy() / 365.25)[:, None]
        Xc = np.column_stack([H, yrs])
        self.temp = {}
        for var in ("tmax", "tmin"):
            y = df[var].to_numpy()
            clim = LinearRegression().fit(Xc, y)
            a = y - clim.predict(Xc)
            Xa = np.column_stack([a[1:-1], a[:-2], wet_anom[2:]])
            ar = LinearRegression().fit(Xa, a[2:])
            res = a[2:] - ar.predict(Xa)
            sd_m = pd.Series(res).groupby(idx.month.to_numpy()[2:]).std().to_dict()
            self.temp[var] = dict(clim=clim, ar=ar, sd=sd_m, last=(a[-1], a[-2]))
        self.last_df = df.tail(10)
        return self

    # ------------------------------------------------------------------ helpers
    def gamma_params(self, doy: int) -> tuple[float, float]:
        if doy not in self._gamma_cache:
            d = np.abs((self.wet_doy - doy + 182) % 365 - 182)
            x = self.wet_amt[d <= 20] - WET_MM + 0.05
            if len(x) < 30:
                x = self.wet_amt - WET_MM + 0.05
            a, _, scale = stats.gamma.fit(x, floc=0)
            self._gamma_cache[doy] = (a, scale)
        return self._gamma_cache[doy]

    def climatology(self, index: pd.DatetimeIndex) -> dict:
        H = harmonics(index)
        yrs = ((index - self.t0).days.to_numpy() / 365.25)[:, None]
        Xc = np.column_stack([H, yrs])
        out = {v: self.temp[v]["clim"].predict(Xc) for v in ("tmax", "tmin")}
        return out

    # ----------------------------------------------------------------- forecast
    def simulate(self, history: pd.DataFrame, dates: pd.DatetimeIndex, n: int,
                 rng: np.random.Generator) -> dict:
        """Ensemble arrays (n, len(dates)) for rain, tmax, tmin."""
        h = history.dropna().tail(10)
        wet_hist = (h["rain"].to_numpy() >= WET_MM).astype(float)
        H = harmonics(dates)
        clim = self.climatology(dates)
        hist_clim = self.climatology(h.index)
        ps = self.pseas.predict_proba(H)[:, 1]
        anom = {v: np.tile((h[v].to_numpy() - hist_clim[v])[-2:][::-1], (n, 1))
                for v in ("tmax", "tmin")}  # columns: lag1, lag2

        T = len(dates)
        rain = np.zeros((n, T))
        out_t = {v: np.zeros((n, T)) for v in ("tmax", "tmin")}
        wet_win = np.tile(wet_hist[-7:], (n, 1))
        prev = np.full(n, wet_hist[-1])
        for j in range(T):
            X = np.column_stack([np.tile(H[j], (n, 1)), prev, wet_win.mean(1),
                                 prev * H[j, 0], prev * H[j, 1]])
            p = self.occ.predict_proba(X)[:, 1]
            wet = (rng.random(n) < p).astype(float)
            a, sc = self.gamma_params(int(dates[j].dayofyear))
            rain[:, j] = wet * (WET_MM + rng.gamma(a, sc, n))
            for v in ("tmax", "tmin"):
                m = self.temp[v]
                Xa = np.column_stack([anom[v][:, 0], anom[v][:, 1], wet - ps[j]])
                new = m["ar"].predict(Xa) + rng.normal(0, m["sd"][dates[j].month], n)
                anom[v] = np.column_stack([new, anom[v][:, 0]])
                out_t[v][:, j] = clim[v][j] + new
            prev = wet
            wet_win = np.column_stack([wet_win[:, 1:], wet])
        out_t["tmin"] = np.minimum(out_t["tmin"], out_t["tmax"] - 1.0)
        return {"rain": rain, **out_t}


def summarize(ens: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    r = ens["rain"]
    return pd.DataFrame({
        "date": dates,
        "rain_mean_mm": r.mean(0), "rain_p10_mm": np.percentile(r, 10, 0),
        "rain_p50_mm": np.percentile(r, 50, 0), "rain_p90_mm": np.percentile(r, 90, 0),
        "rain_prob_pct": (r >= WET_MM).mean(0) * 100,
        "heavy_rain_prob_pct": (r >= 35.1).mean(0) * 100,  # TMD "heavy rain" threshold
        "tmax_mean_c": ens["tmax"].mean(0), "tmax_p10_c": np.percentile(ens["tmax"], 10, 0),
        "tmax_p90_c": np.percentile(ens["tmax"], 90, 0),
        "tmin_mean_c": ens["tmin"].mean(0), "tmin_p10_c": np.percentile(ens["tmin"], 10, 0),
        "tmin_p90_c": np.percentile(ens["tmin"], 90, 0),
    })
