import os
import logging
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from .config import CFG

def setup_logging(logdir: str, script_name: str) -> str:
    os.makedirs(logdir, exist_ok=True)
    log_path = os.path.join(logdir, f"{script_name}.log")
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized: %s", log_path)
    return log_path

def _find_timestamp_col(df: pd.DataFrame) -> str:
    for cand in CFG.TIMESTAMP_CANDIDATES:
        if cand in df.columns:
            return cand
    # fallback: first column
    return df.columns[0]

def read_any(data_path: str) -> pd.DataFrame:
    """
    Read CSV or Excel with robust datetime parsing.
    - If multiple sheets, read the first.
    - Whitespace/surrounding columns are trimmed.
    """
    ext = os.path.splitext(data_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(data_path)  # requires openpyxl/xlrd
    else:
        df = pd.read_csv(data_path)
    # Trim column names
    df.columns = [str(c).strip() for c in df.columns]
    ts_col = _find_timestamp_col(df)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", infer_datetime_format=True, utc=False)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    # Coerce other columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def enforce_5min_grid(df: pd.DataFrame, resample_rule: str = None) -> pd.DataFrame:
    """
    Resample to strict grid (default 5‑min). Short gaps are interpolated (up to CFG.MAX_INTERP_MIN).
    Longer gaps remain NaN.
    """
    rule = resample_rule or CFG.RESAMPLE_RULE
    out = df.resample(rule).mean()
    # Interpolate short gaps per column
    limit = max(1, CFG.MAX_INTERP_MIN // int(rule.replace("min","")))
    out = out.interpolate(method="time", limit=limit, limit_direction="both")
    return out

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T

def mad_clip_series(s: pd.Series, k: float) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s  # nothing to clip
    lower = med - k * mad
    upper = med + k * mad
    return s.clip(lower, upper)

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def savefig(path: str, fig):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logging.info("Saved figure: %s", path)

def rolling_groups_to_periods(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Convert boolean mask over a DatetimeIndex to contiguous (start, end) periods.
    """
    idx = mask.index
    periods = []
    if mask.empty:
        return periods
    run_start = None
    for i in range(len(mask)):
        val = bool(mask.iloc[i])
        if val and run_start is None:
            run_start = idx[i]
        if run_start is not None and (not val or i == len(mask)-1):
            # close run
            end_idx = idx[i] if val and i == len(mask)-1 else idx[i-1]
            periods.append((run_start, end_idx))
            run_start = None
    return periods

def periods_to_df(periods: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    rows = []
    for s,e in periods:
        dur_min = (e - s).total_seconds() / 60.0
        rows.append({"start": s, "end": e, "duration_min": dur_min})
    return pd.DataFrame(rows)

def dedup_overlapping_periods(df_periods: pd.DataFrame) -> pd.DataFrame:
    if df_periods.empty:
        return df_periods
    df = df_periods.sort_values("start").reset_index(drop=True)
    merged = []
    cur_s, cur_e = df.loc[0, "start"], df.loc[0, "end"]
    for i in range(1, len(df)):
        s, e = df.loc[i, "start"], df.loc[i, "end"]
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    out = periods_to_df(merged)
    return out
