"""Thailand weather & dam-storage forecast, 24-30 Sep 2026.

    python run_forecast.py                 # auto: observed CSV > Open-Meteo > synthetic
    python run_forecast.py --source synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from thai_forecast import config as C  # noqa: E402
from thai_forecast.dam_model import DamModel, summarize as dam_summary  # noqa: E402
from thai_forecast.data import load_all  # noqa: E402
from thai_forecast.plots import plot_all  # noqa: E402
from thai_forecast.weather_model import WET_MM, WeatherModel, summarize as wx_summary  # noqa: E402

OUT = Path(__file__).parent / "outputs"
TRAIN_END = "2015-12-31"
BT_YEARS = range(2016, 2026)
BT_ISSUE_DAYS = range(1, 24, 2)  # issue dates 1,3,...,23 September
BT_MEMBERS = 300
H = 7


def backtest(weather, dams, rng):
    """Hindcast 7-day forecasts issued in September 2016-2025 (models trained on 2000-2015)."""
    rows = []
    wms = {k: WeatherModel().fit(df.loc[:TRAIN_END]) for k, df in weather.items()}
    dms = {d.key: DamModel(d).fit(dams[d.key].loc[:TRAIN_END], weather[d.station]["rain"].loc[:TRAIN_END])
           for d in C.DAMS}
    for k, df in weather.items():
        train = df.loc[:TRAIN_END]
        sep = train[train.index.month == 9]
        clim_p = (sep["rain"] >= WET_MM).mean()
        clim_tot = sep["rain"].mean() * H
        for y in BT_YEARS:
            for d0 in BT_ISSUE_DAYS:
                issue = pd.Timestamp(y, 9, d0)
                dates = pd.date_range(issue, periods=H)
                obs = df.loc[dates]
                ens = wms[k].simulate(df.loc[:issue - pd.Timedelta(days=1)], dates, BT_MEMBERS, rng)
                clim = wms[k].climatology(dates)
                wet_o = (obs["rain"].to_numpy() >= WET_MM).astype(float)
                p = (ens["rain"] >= WET_MM).mean(0)
                row = dict(kind="weather", site=k, issue=issue,
                           tmax_mae=np.abs(ens["tmax"].mean(0) - obs["tmax"]).mean(),
                           tmax_mae_clim=np.abs(clim["tmax"] - obs["tmax"]).mean(),
                           tmin_mae=np.abs(ens["tmin"].mean(0) - obs["tmin"]).mean(),
                           tmin_mae_clim=np.abs(clim["tmin"] - obs["tmin"]).mean(),
                           brier=((p - wet_o) ** 2).mean(), brier_clim=((clim_p - wet_o) ** 2).mean(),
                           rain7_ae=abs(ens["rain"].sum(1).mean() - obs["rain"].sum()),
                           rain7_ae_clim=abs(clim_tot - obs["rain"].sum()))
                rows.append(row)
                for dam in (d for d in C.DAMS if d.station == k):
                    hist = dams[dam.key].loc[:issue - pd.Timedelta(days=1)]
                    de = dms[dam.key].simulate(hist, df["rain"], ens["rain"], dates, rng)
                    o = dams[dam.key].loc[dates]
                    rows.append(dict(
                        kind="dam", site=dam.key, issue=issue,
                        storage_ae=abs(de["storage"][:, -1].mean() - o["storage"].iloc[-1]),
                        storage_ae_persist=abs(hist["storage"].iloc[-1] - o["storage"].iloc[-1]),
                        inflow7_ae=abs(de["inflow"].sum(1).mean() - o["inflow"].sum()),
                        inflow7_ae_persist=abs(hist["inflow"].iloc[-1] * H - o["inflow"].sum()),
                        in_p10_p90=float(np.percentile(de["storage"][:, -1], 10) <= o["storage"].iloc[-1]
                                         <= np.percentile(de["storage"][:, -1], 90)),
                        capacity=dam.capacity_mcm))
    bt = pd.DataFrame(rows)
    w = bt[bt.kind == "weather"].groupby("site").mean(numeric_only=True)
    w = w[["tmax_mae", "tmax_mae_clim", "tmin_mae", "tmin_mae_clim", "brier", "brier_clim",
           "rain7_ae", "rain7_ae_clim"]]
    w["tmax_skill_pct"] = (1 - w.tmax_mae / w.tmax_mae_clim) * 100
    w["tmin_skill_pct"] = (1 - w.tmin_mae / w.tmin_mae_clim) * 100
    w["brier_skill_pct"] = (1 - w.brier / w.brier_clim) * 100
    w["rain7_skill_pct"] = (1 - w.rain7_ae / w.rain7_ae_clim) * 100
    d = bt[bt.kind == "dam"].groupby("site").mean(numeric_only=True)
    d = d[["storage_ae", "storage_ae_persist", "inflow7_ae", "inflow7_ae_persist", "in_p10_p90", "capacity"]]
    d["storage_ae_pct_cap"] = d.storage_ae / d.capacity * 100
    d["storage_skill_pct"] = (1 - d.storage_ae / d.storage_ae_persist) * 100
    d["inflow7_skill_pct"] = (1 - d.inflow7_ae / d.inflow7_ae_persist) * 100
    d["coverage_p10_p90_pct"] = d.in_p10_p90 * 100
    return w.round(3), d.drop(columns=["in_p10_p90"]).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=["auto", "openmeteo", "synthetic"])
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(C.SEED)

    weather, dams, prov = load_all(args.source)
    print("data sources:", prov)
    dates = pd.date_range(C.FORECAST_START, C.FORECAST_END)

    # ---- fit on full history and forecast
    wx_rows, dam_rows, fits, ens_store = [], [], {}, {}
    for st in C.STATIONS:
        m = WeatherModel().fit(weather[st.key])
        ens = m.simulate(weather[st.key], dates, C.N_ENSEMBLE, rng)
        ens_store[st.key] = ens
        s = wx_summary(ens, dates)
        clim = m.climatology(dates)
        s.insert(0, "station", st.name)
        s.insert(1, "region", st.region)
        s["tmax_clim_c"], s["tmin_clim_c"] = clim["tmax"], clim["tmin"]
        wx_rows.append(s)
        fits[st.key] = dict(occ_coef=m.occ.coef_[0].round(3).tolist(),
                            gamma_shape_scale=[round(v, 3) for v in m.gamma_params(int(dates[3].dayofyear))],
                            tmax_ar=m.temp["tmax"]["ar"].coef_.round(3).tolist(),
                            tmin_ar=m.temp["tmin"]["ar"].coef_.round(3).tolist())
    for dam in C.DAMS:
        dm = DamModel(dam).fit(dams[dam.key], weather[dam.station]["rain"])
        ens = dm.simulate(dams[dam.key], weather[dam.station]["rain"], ens_store[dam.station]["rain"],
                          dates, rng)
        s = dam_summary(dam, ens, dates)
        s.insert(0, "dam", dam.name)
        s.insert(1, "province", dam.province)
        s["capacity_mcm"] = dam.capacity_mcm
        dam_rows.append(s)
        fits[dam.key] = dict(inflow_r2=round(dm.inflow_r2, 3), release_r2=round(dm.release_r2, 3),
                             inflow_coef=dm.inflow.coef_[:5].round(3).tolist(), res_phi=round(dm.res_phi, 3))
    wx, dm_df = pd.concat(wx_rows), pd.concat(dam_rows)
    for df in (wx, dm_df):
        num = df.select_dtypes("number").columns
        df[num] = df[num].round(2)
    wx.to_csv(OUT / "forecast_weather.csv", index=False)
    dm_df.to_csv(OUT / "forecast_dams.csv", index=False)

    bt_w = bt_d = None
    if not args.no_backtest:
        bt_w, bt_d = backtest(weather, dams, np.random.default_rng(C.SEED + 1))
        bt_w.to_csv(OUT / "backtest_weather.csv")
        bt_d.to_csv(OUT / "backtest_dams.csv")
        print(bt_w[["tmax_skill_pct", "tmin_skill_pct", "brier_skill_pct", "rain7_skill_pct"]])
        print(bt_d[["storage_ae", "storage_ae_pct_cap", "storage_skill_pct", "coverage_p10_p90_pct"]])

    # ---- JSON bundle for the HTML report
    hist_days = 60
    bundle = dict(
        window=[C.FORECAST_START, C.FORECAST_END], members=C.N_ENSEMBLE, provenance=prov,
        stations=[dict(key=s.key, name=s.name, region=s.region,
                       history=weather[s.key].tail(14).reset_index().assign(
                           date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records"),
                       forecast=wx[wx.station == s.name].assign(
                           date=lambda x: x.date.dt.strftime("%Y-%m-%d")).drop(
                           columns=["station", "region"]).to_dict("records"),
                       fit=fits[s.key]) for s in C.STATIONS],
        dams=[dict(key=d.key, name=d.name, province=d.province, capacity=d.capacity_mcm, dead=d.dead_mcm,
                   history=dams[d.key].tail(hist_days).reset_index().assign(
                       date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records"),
                   forecast=dm_df[dm_df.dam == d.name].assign(
                       date=lambda x: x.date.dt.strftime("%Y-%m-%d")).drop(
                       columns=["dam", "province"]).to_dict("records"),
                   fit=fits[d.key]) for d in C.DAMS],
        backtest=dict(weather=None if bt_w is None else bt_w.reset_index().to_dict("records"),
                      dams=None if bt_d is None else bt_d.reset_index().to_dict("records")),
    )
    (OUT / "forecast.json").write_text(json.dumps(bundle, indent=1, default=float))
    plot_all(weather, dams, wx, dm_df, OUT / "figures")
    print(wx[["station", "date", "rain_mean_mm", "rain_prob_pct", "tmax_mean_c", "tmin_mean_c"]].to_string())
    print(dm_df[["dam", "date", "storage_mean_mcm", "storage_pct_capacity", "inflow_mean_mcm"]].to_string())


if __name__ == "__main__":
    main()
