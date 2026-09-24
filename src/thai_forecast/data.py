"""Data loading.

Three sources, tried in order:

1. ``data/observed/`` - user-supplied CSVs (real observations always win):
   * ``weather_<station>.csv`` with columns ``date,rain,tmax,tmin``
   * ``dam_<dam>.csv`` with columns ``date,storage,inflow,release``  (MCM)
2. Open-Meteo historical archive (ERA5) for weather, if the network allows it.
3. An offline stochastic history generated from TMD monthly normals and dam
   physical parameters (``synthetic``). This keeps the whole pipeline runnable
   anywhere, but its numbers are *illustrative*, not observations.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parents[2]
OBS_DIR = ROOT / "data" / "observed"
CACHE_DIR = ROOT / "data" / "cache"


# --------------------------------------------------------------------------- utils
def daily_climatology(monthly: tuple, index: pd.DatetimeIndex) -> np.ndarray:
    """Periodic linear interpolation of 12 mid-month values onto daily dates."""
    mid = np.array([15.5 + 30.44 * m for m in range(12)])
    doy = index.dayofyear.to_numpy().astype(float)
    xp = np.concatenate([mid - 365.25, mid, mid + 365.25])
    fp = np.tile(np.asarray(monthly, float), 3)
    return np.interp(doy, xp, fp)


def history_end() -> pd.Timestamp:
    return pd.Timestamp(C.FORECAST_START) - pd.Timedelta(days=1)


# ---------------------------------------------------------------- live sources
def _open_meteo(st: C.Station, start: str, end: str) -> pd.DataFrame | None:
    try:
        import requests
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=dict(latitude=st.lat, longitude=st.lon, start_date=start, end_date=end,
                        daily="precipitation_sum,temperature_2m_max,temperature_2m_min",
                        timezone="Asia/Bangkok"),
            timeout=30)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame({"date": pd.to_datetime(d["time"]), "rain": d["precipitation_sum"],
                           "tmax": d["temperature_2m_max"], "tmin": d["temperature_2m_min"]})
        return df.set_index("date").interpolate(limit=3).dropna()
    except Exception:
        return None


# ------------------------------------------------------------ synthetic weather
def synth_weather(st: C.Station, rng: np.random.Generator) -> pd.DataFrame:
    idx = pd.date_range(C.HISTORY_START, history_end(), freq="D")
    n = len(idx)
    dim = idx.days_in_month.to_numpy()
    wet_frac = np.clip(daily_climatology(st.wet_days, idx) / dim, 0.02, 0.95)
    wet_mean = daily_climatology(np.array(st.rain_mm) / np.maximum(st.wet_days, 1), idx)
    tmax_c = daily_climatology(st.tmax, idx)
    tmin_c = daily_climatology(st.tmin, idx)

    years = idx.year.to_numpy()
    uy = np.unique(years)
    rain_mult = dict(zip(uy, rng.lognormal(0, 0.18, len(uy))))  # ENSO-like wet/dry years
    t_anom = dict(zip(uy, rng.normal(0, 0.35, len(uy))))
    trend = 0.02 * (years - 2000)

    r_persist, shape = 0.35, 0.75
    wet = np.zeros(n, bool)
    rain = np.zeros(n)
    ax = an = 0.0
    tmax = np.zeros(n)
    tmin = np.zeros(n)
    for t in range(n):
        pi = wet_frac[t]
        p11 = pi + r_persist * (1 - pi)
        p01 = pi * (1 - p11) / (1 - pi)
        p = p11 if (t and wet[t - 1]) else p01
        wet[t] = rng.random() < min(0.97, p * np.sqrt(rain_mult[years[t]]))
        if wet[t]:
            rain[t] = 1.0 + rng.gamma(shape, max(wet_mean[t] * rain_mult[years[t]] - 1, 0.5) / shape)
        ax = 0.70 * ax + rng.normal(0, 1.1)
        an = 0.75 * an + rng.normal(0, 0.7)
        yt = t_anom[years[t]] + trend[t]
        tmax[t] = tmax_c[t] + yt + ax - (1.4 if wet[t] else -0.4)
        tmin[t] = tmin_c[t] + yt + an + (0.2 if wet[t] else -0.1)
    tmin = np.minimum(tmin, tmax - 2.0)
    return pd.DataFrame({"rain": rain.round(1), "tmax": tmax.round(1), "tmin": tmin.round(1)},
                        index=pd.Index(idx, name="date"))


# --------------------------------------------------------------- dam physics
def catchment_rain(station_rain: np.ndarray) -> np.ndarray:
    """Areal-average proxy: blend of point rain and its trailing 3-day mean."""
    r = np.asarray(station_rain, float)
    m3 = pd.Series(r).rolling(3, min_periods=1).mean().to_numpy()
    return 0.4 * r + 0.6 * m3


def antecedent_index(p: np.ndarray, k: float = 0.93, api0: float = 0.0) -> np.ndarray:
    out = np.empty_like(p, dtype=float)
    a = api0
    for i, v in enumerate(p):
        a = k * a + v
        out[i] = a
    return out


def rule_target(dam: C.Dam, index: pd.DatetimeIndex) -> np.ndarray:
    return daily_climatology(dam.rule_curve, index) * dam.capacity_mcm


def evaporation(dam: C.Dam, storage: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    e = daily_climatology(C.EVAP_MM_DAY, index)
    area = dam.surface_km2 * np.clip(storage / dam.capacity_mcm, 0, 1.1) ** 0.7
    return e * area * 1e-3  # mm * km2 * 1e-3 = MCM


def synth_dam(dam: C.Dam, weather: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    idx = weather.index
    p = catchment_rain(weather["rain"].to_numpy())
    api = antecedent_index(p)
    eff = p * api / (api + 150.0)  # wetter catchment -> higher runoff coefficient
    kq, ks = np.exp(-1 / 6.0), np.exp(-1 / 60.0)
    q = np.zeros(len(p))
    fq = sq = 0.0
    for t, e in enumerate(eff):
        fq = kq * fq + (1 - kq) * 0.7 * e
        sq = ks * sq + (1 - ks) * 0.3 * e
        q[t] = fq + sq
    inflow = q * dam.catchment_km2 * 1e-3
    inflow *= dam.annual_inflow_mcm / (inflow.mean() * 365.25)
    inflow *= rng.lognormal(0, 0.10, len(inflow))

    target = rule_target(dam, idx)
    base = dam.annual_inflow_mcm / 365.25 * 0.92
    rmin, rmax = 0.15 * base, 4.0 * base
    e_rate = daily_climatology(C.EVAP_MM_DAY, idx)
    s = target[0]
    rel_prev = base
    storage = np.zeros(len(p))
    release = np.zeros(len(p))
    for t in range(len(p)):
        want = base + 0.02 * (s - target[t])
        rel = np.clip(0.8 * rel_prev + 0.2 * want + rng.normal(0, 0.03 * base), rmin, rmax)
        ev = e_rate[t] * dam.surface_km2 * min(s / dam.capacity_mcm, 1.1) ** 0.7 * 1e-3
        s_new = s + inflow[t] - rel - ev
        if s_new > dam.capacity_mcm * 1.02:  # spillway
            rel += s_new - dam.capacity_mcm * 1.02
            s_new = dam.capacity_mcm * 1.02
        if s_new < dam.dead_mcm:
            rel = max(rel - (dam.dead_mcm - s_new), 0)
            s_new = dam.dead_mcm
        storage[t], release[t], s, rel_prev = s_new, rel, s_new, rel
    return pd.DataFrame({"storage": storage, "inflow": inflow, "release": release},
                        index=idx).round(2)


# ------------------------------------------------------------------ public API
def load_all(source: str = "auto") -> tuple[dict, dict, dict]:
    """Return (weather, dams, provenance). ``source``: auto | openmeteo | synthetic."""
    rng = np.random.default_rng(C.SEED)
    weather, dams, prov = {}, {}, {}
    end = history_end().strftime("%Y-%m-%d")
    for st in C.STATIONS:
        obs = OBS_DIR / f"weather_{st.key}.csv"
        df = None
        if obs.exists():
            df, prov[st.key] = pd.read_csv(obs, parse_dates=["date"], index_col="date"), "observed"
        elif source in ("auto", "openmeteo"):
            df = _open_meteo(st, C.HISTORY_START, end)
            if df is not None:
                prov[st.key] = "open-meteo"
        synth = synth_weather(st, rng)  # always drawn so the RNG stream is stable
        if df is None:
            df, prov[st.key] = synth, "synthetic"
        weather[st.key] = df
    for dam in C.DAMS:
        obs = OBS_DIR / f"dam_{dam.key}.csv"
        synth = synth_dam(dam, weather[dam.station], rng)
        if obs.exists():
            dams[dam.key], prov[dam.key] = pd.read_csv(obs, parse_dates=["date"], index_col="date"), "observed"
        else:
            dams[dam.key], prov[dam.key] = synth, "synthetic"
    return weather, dams, prov
