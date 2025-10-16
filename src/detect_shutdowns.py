import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import CFG
from .utils_io import rolling_groups_to_periods, periods_to_df, dedup_overlapping_periods, ensure_outdir, savefig

def detect_shutdown_mask(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic rule‑based detection:
    - Smooth key sensors with rolling median over IDLE_WINDOW_MIN.
    - Define shutdown if *all* available sensor values are below thresholds.
      (Strict AND → conservative; change to majority vote if needed.)
    - Enforce minimum duration CFG.MIN_DURATION_MIN.
    """
    df_s = df.copy()
    win = max(1, CFG.IDLE_WINDOW_MIN // int(CFG.RESAMPLE_RULE.replace("min","")))
    for c in CFG.SENSOR_COLS:
        if c in df_s.columns:
            df_s[c] = df_s[c].rolling(win, center=True, min_periods=max(1, win//2)).median()

    below = []
    for c, thr in CFG.SHUTDOWN_THRESHOLDS.items():
        if c in df_s.columns:
            below.append(df_s[c] <= thr)
    if not below:
        # If no known cols present, fallback: target only
        s = df_s[CFG.TARGET_COL] if CFG.TARGET_COL in df_s else pd.Series(False, index=df_s.index)
        below = [s <= s.quantile(0.1)]  # bottom decile as heuristic

    mask_raw = below[0]
    for m in below[1:]:
        mask_raw = mask_raw & m  # conservative AND

    # Enforce min duration
    periods = rolling_groups_to_periods(mask_raw)
    min_steps = max(1, CFG.MIN_DURATION_MIN // int(CFG.RESAMPLE_RULE.replace("min","")))
    filtered = []
    for s,e in periods:
        if (e - s).total_seconds() / 60.0 >= CFG.MIN_DURATION_MIN:
            filtered.append((s,e))
    mask = pd.Series(False, index=df.index)
    for s,e in filtered:
        mask.loc[s:e] = True
    return mask

def run_shutdown_detection(df: pd.DataFrame, outdir: str):
    ensure_outdir(outdir)
    mask = detect_shutdown_mask(df)
    periods = rolling_groups_to_periods(mask)
    dfp = periods_to_df(periods)
    dfp["reason"] = "below_thresholds"
    dfp = dedup_overlapping_periods(dfp)

    # Stats
    total_dt = dfp["duration_min"].sum() if not dfp.empty else 0.0
    logging.info("Detected %d shutdown/idle periods; total downtime = %.1f min", len(dfp), total_dt)

    # Save CSV
    csv_path = os.path.join(outdir, "shutdown_periods.csv")
    dfp.to_csv(csv_path, index=False)
    logging.info("Saved shutdown periods: %s", csv_path)

    # Year plot with shutdowns highlighted (overlay on target)
    if CFG.TARGET_COL in df.columns and len(df) > 0:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df.index, df[CFG.TARGET_COL], label=CFG.TARGET_COL, linewidth=0.8)
        for _, row in dfp.iterrows():
            ax.axvspan(row["start"], row["end"], color="red", alpha=0.15)
        ax.set_title("Shutdowns (highlighted) over time")
        ax.set_xlabel("Time"); ax.set_ylabel("Value"); ax.legend()
        savefig(os.path.join(outdir, "shutdown_plot.png"), fig); plt.close(fig)

    return dfp
