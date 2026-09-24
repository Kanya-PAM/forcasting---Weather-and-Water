"""Static PNG figures (matplotlib) for the repository report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_all(weather, dams, wx: pd.DataFrame, dm: pd.DataFrame, out: Path):
    start = pd.Timestamp(C.FORECAST_START)

    # --- weather: one row per station, rain (left) and temperature (right)
    fig, axes = plt.subplots(len(C.STATIONS), 2, figsize=(11, 2.3 * len(C.STATIONS)), sharex=True)
    for i, st in enumerate(C.STATIONS):
        h = weather[st.key].tail(14)
        f = wx[wx.station == st.name]
        ax = axes[i, 0]
        ax.bar(h.index, h["rain"], color="#9ec5f4", width=0.8, label="history")
        ax.bar(f["date"], f["rain_mean_mm"], color=BLUE, width=0.8, label="forecast mean")
        ax.vlines(f["date"], f["rain_p10_mm"], f["rain_p90_mm"], color=INK, lw=1, label="P10-P90")
        ax.set_ylabel(f"{st.name}\nrain (mm)", color=INK, fontsize=9)
        _style(ax)
        ax = axes[i, 1]
        ax.plot(h.index, h["tmax"], color=ORANGE, lw=1, alpha=0.5)
        ax.plot(h.index, h["tmin"], color=BLUE, lw=1, alpha=0.5)
        ax.fill_between(f["date"], f["tmax_p10_c"], f["tmax_p90_c"], color=ORANGE, alpha=0.18, lw=0)
        ax.fill_between(f["date"], f["tmin_p10_c"], f["tmin_p90_c"], color=BLUE, alpha=0.18, lw=0)
        ax.plot(f["date"], f["tmax_mean_c"], color=ORANGE, lw=2, label="Tmax")
        ax.plot(f["date"], f["tmin_mean_c"], color=BLUE, lw=2, label="Tmin")
        ax.set_ylabel("temp (°C)", color=INK, fontsize=9)
        _style(ax)
        for a in axes[i]:
            a.axvline(start - pd.Timedelta(hours=12), color=MUTED, lw=0.8, ls="--")
        if i == 0:
            axes[0, 0].legend(fontsize=7, frameon=False, ncol=3)
            axes[0, 1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Weather forecast 24–30 Sep 2026 (dashed line = forecast start)", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "weather_forecast.png", dpi=130)
    plt.close(fig)

    # --- dams: storage % capacity, last 60 days + forecast fan
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, d in zip(axes.flat, C.DAMS):
        h = dams[d.key].tail(60)
        f = dm[dm.dam == d.name]
        cap = d.capacity_mcm
        ax.plot(h.index, h["storage"] / cap * 100, color=INK, lw=1.2, label="history")
        ax.fill_between(f["date"], f["storage_p10_mcm"] / cap * 100, f["storage_p90_mcm"] / cap * 100,
                        color=BLUE, alpha=0.25, lw=0, label="P10–P90")
        ax.plot(f["date"], f["storage_pct_capacity"], color=BLUE, lw=2, label="forecast mean")
        ax.axhline(100, color=ORANGE, lw=1, ls=":")
        ax.set_title(f"{d.name} ({d.province}) – cap {cap:,.0f} MCM", fontsize=9, color=INK)
        ax.set_ylabel("% of capacity", fontsize=8, color=MUTED)
        _style(ax)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Dam storage forecast 24–30 Sep 2026 (dotted = 100 % capacity)", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "dam_storage_forecast.png", dpi=130)
    plt.close(fig)
