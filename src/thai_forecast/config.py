"""Static configuration: forecast window, stations and dams.

Monthly climate normals are approximate 1991-2020 values from the Thai
Meteorological Department (TMD). Dam figures are approximate published values
from EGAT / the Royal Irrigation Department (RID). All are rounded; they are
used to seed the offline history generator and as physical parameters of the
water-balance model.
"""
from dataclasses import dataclass, field

FORECAST_START = "2026-09-24"
FORECAST_END = "2026-09-30"
HISTORY_START = "2000-01-01"
N_ENSEMBLE = 1000
SEED = 20260924


@dataclass(frozen=True)
class Station:
    key: str
    name: str
    region: str
    lat: float
    lon: float
    rain_mm: tuple  # monthly totals Jan..Dec
    wet_days: tuple  # monthly count of days with rain >= 1 mm
    tmax: tuple  # monthly mean daily max (degC)
    tmin: tuple  # monthly mean daily min (degC)


STATIONS = [
    Station("chiangmai", "Chiang Mai", "North", 18.79, 98.98,
            (5, 6, 14, 50, 160, 130, 160, 230, 230, 120, 50, 15),
            (1, 1, 2, 5, 14, 17, 20, 22, 18, 11, 4, 1),
            (29.6, 32.5, 35.3, 36.4, 34.1, 32.3, 31.4, 31.0, 31.4, 31.0, 29.6, 28.3),
            (14.3, 15.4, 18.4, 22.0, 23.5, 23.8, 23.4, 23.2, 22.8, 21.5, 18.6, 15.2)),
    Station("khonkaen", "Khon Kaen", "Northeast", 16.43, 102.83,
            (5, 18, 40, 80, 190, 190, 190, 230, 250, 110, 15, 3),
            (1, 2, 4, 7, 14, 16, 17, 19, 17, 9, 2, 1),
            (30.7, 33.2, 35.6, 36.4, 34.7, 33.5, 32.8, 32.2, 31.9, 31.5, 30.6, 29.6),
            (17.2, 19.8, 22.9, 24.8, 25.0, 25.0, 24.6, 24.3, 24.0, 22.6, 19.8, 17.1)),
    Station("bangkok", "Bangkok", "Central", 13.75, 100.50,
            (13, 20, 42, 92, 222, 157, 175, 219, 334, 292, 49, 7),
            (2, 2, 4, 6, 16, 16, 18, 20, 21, 17, 5, 1),
            (32.5, 33.3, 34.3, 35.4, 34.4, 33.6, 33.2, 32.9, 32.6, 32.4, 32.1, 31.6),
            (22.0, 23.8, 25.4, 26.6, 26.3, 26.0, 25.6, 25.5, 25.1, 24.9, 23.6, 21.6)),
    Station("kanchanaburi", "Kanchanaburi", "West", 14.02, 99.53,
            (6, 15, 35, 75, 160, 110, 130, 140, 250, 210, 40, 5),
            (1, 2, 3, 6, 13, 14, 16, 17, 18, 13, 4, 1),
            (32.7, 35.2, 37.3, 38.1, 36.2, 34.2, 33.4, 33.0, 32.6, 32.0, 31.4, 31.1),
            (18.6, 20.9, 23.5, 25.2, 25.4, 25.1, 24.8, 24.6, 24.2, 23.5, 21.3, 18.6)),
    Station("chanthaburi", "Chanthaburi", "East", 12.61, 102.10,
            (30, 50, 90, 150, 400, 550, 550, 570, 480, 240, 60, 15),
            (2, 3, 6, 10, 20, 24, 24, 25, 23, 16, 5, 2),
            (32.4, 32.6, 33.0, 33.4, 32.3, 30.9, 30.4, 30.3, 30.9, 31.6, 32.1, 32.0),
            (21.2, 22.6, 23.9, 24.6, 24.7, 24.4, 24.0, 23.9, 23.6, 23.2, 22.0, 20.8)),
    Station("suratthani", "Surat Thani", "South", 9.13, 99.33,
            (80, 30, 40, 80, 170, 130, 140, 140, 170, 300, 440, 200),
            (8, 4, 5, 8, 16, 15, 16, 17, 18, 21, 19, 12),
            (30.5, 32.0, 33.6, 34.7, 34.3, 33.5, 33.1, 33.1, 32.6, 31.3, 29.9, 29.6),
            (21.8, 21.9, 22.6, 23.4, 23.9, 23.6, 23.2, 23.2, 23.1, 22.9, 22.6, 22.1)),
]
STATION_BY_KEY = {s.key: s for s in STATIONS}


@dataclass(frozen=True)
class Dam:
    key: str
    name: str
    province: str
    station: str  # key of the station used as catchment-rain proxy
    capacity_mcm: float  # normal-high-water storage
    dead_mcm: float  # dead storage
    catchment_km2: float
    surface_km2: float  # reservoir surface area at full storage
    annual_inflow_mcm: float  # long-term mean annual inflow
    # rule-curve target storage as fraction of capacity, Jan..Dec
    rule_curve: tuple = field(default=(0.66, 0.62, 0.57, 0.52, 0.48, 0.47,
                                       0.50, 0.57, 0.66, 0.74, 0.76, 0.72))


DAMS = [
    Dam("bhumibol", "Bhumibol", "Tak", "chiangmai", 13462, 3800, 26386, 300, 5700),
    Dam("sirikit", "Sirikit", "Uttaradit", "chiangmai", 9510, 2850, 13130, 250, 5800),
    Dam("ubolratana", "Ubol Ratana", "Khon Kaen", "khonkaen", 2431, 581, 12000, 410, 2500),
    Dam("srinagarind", "Srinagarind", "Kanchanaburi", "kanchanaburi", 17745, 7470, 10880, 419, 4900),
    Dam("pasakjolasid", "Pasak Jolasid", "Lopburi", "bangkok", 960, 3, 14520, 170, 2300,
        rule_curve=(0.40, 0.33, 0.27, 0.22, 0.18, 0.17, 0.22, 0.40, 0.70, 0.90, 0.85, 0.60)),
    Dam("rajjaprabha", "Rajjaprabha", "Surat Thani", "suratthani", 5639, 1352, 1435, 165, 2400),
]

# Monthly open-water evaporation (mm/day), a typical Thai pan value x 0.8
EVAP_MM_DAY = (3.6, 4.2, 4.8, 5.0, 4.4, 3.8, 3.6, 3.4, 3.3, 3.4, 3.4, 3.4)
