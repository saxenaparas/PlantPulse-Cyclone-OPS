"""
All project‑wide constants and sensible defaults live here.
Tune thresholds here (or override via CLI flags where supported).

If any constant was ambiguous in the spec, we chose a practical default
and documented it inline below.
"""

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Config:
    # --- Reproducibility ---
    RANDOM_SEED: int = 42  # deterministic seeds everywhere

    # --- Input/Parsing ---
    TIMESTAMP_CANDIDATES: List[str] = field(default_factory=lambda: [
        "timestamp", "time", "datetime", "date", "Timestamp", "DateTime"
    ])
    RESAMPLE_RULE: str = "5min"  # strict 5‑minute grid
    MAX_INTERP_MIN: int = 60     # interpolate gaps up to 60 min; else leave NaN

    # --- Columns ---
    # Default target variable (spec's "forecast Cyclone_Inlet_Gas_Temp")
    TARGET_COL: str = "Cyclone_Inlet_Gas_Temp"
    # Expected core sensor cols (used for summaries/correlations if present)
    SENSOR_COLS: List[str] = field(default_factory=lambda: [
        "Cyclone_Inlet_Gas_Temp",
        "Cyclone_Gas_Outlet_Temp",
        "Cyclone_Outlet_Gas_draft",
        "Cyclone_cone_draft",
        "Cyclone_Inlet_Draft",
        "Cyclone_Material_Temp",
    ])

    # --- EDA / Outliers ---
    # Robust MAD clipping: clip to median ± MAD_CLIP_K * MAD; chosen default = 5.0 (conservative)
    MAD_CLIP_K: float = 5.0

    # --- Shutdown detection (rule‑based) ---
    # Ambiguity: no explicit thresholds provided → choose practical defaults in engineering units.
    # We assume shutdown/idle is characterized by *low drafts/pressures* and *low temperature*.
    # Thresholds are applied after optional smoothing with rolling median (IDLE_WINDOW_MIN).
    SHUTDOWN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "Cyclone_Inlet_Gas_Temp": 80.0,        # deg C (example conservative low)
        "Cyclone_Gas_Outlet_Temp": 80.0,       # deg C
        "Cyclone_Outlet_Gas_draft": 5.0,       # pressure units
        "Cyclone_cone_draft": 5.0,
        "Cyclone_Inlet_Draft": 5.0,
        "Cyclone_Material_Temp": 60.0,
    })
    # Minimum continuous duration (minutes) below thresholds to flag shutdown/idle
    MIN_DURATION_MIN: int = 30
    # Rolling window for idle smoothing (minutes)
    IDLE_WINDOW_MIN: int = 20

    # --- Clustering ---
    # Rolling window sizes (minutes) for features; with 5‑min grid these are 3/6/12 points.
    FEAT_WINDOWS_MIN: List[int] = field(default_factory=lambda: [15, 30, 60])
    KMIN: int = 2
    KMAX: int = 5
    PCA_COMPONENTS: int = 2

    # --- Anomalies ---
    ISO_N_ESTIMATORS: int = 200
    ISO_MAX_SAMPLES: str = "auto"
    ISO_CONTAM: float = 0.01  # expected anomaly fraction; conservative
    MAD_Z_K: float = 6.0      # rolling-MAD fallback threshold (6 robust-sigmas)

    # --- Forecasting ---
    HORIZON_STEPS: int = 12   # 12×5min = 1 hour ahead
    DAILY_STEPS: int = 288    # 24h at 5‑min resolution → for seasonal naive & ACF checks
    ARIMA_P: List[int] = field(default_factory=lambda: [0,1,2])
    ARIMA_D: List[int] = field(default_factory=lambda: [0,1])
    ARIMA_Q: List[int] = field(default_factory=lambda: [0,1,2])
    GB_LAGS: int = 36         # 3 hours of lags (sensible for short‑horizon; deterministic)
    BACKTEST_DAYS: int = 30   # rolling backtest over the last 30 days by default
    MIN_TRAIN_DAYS: int = 60  # minimum training span before first forecast window

CFG = Config()

