import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

from .config import CFG
from .features import build_rolling_features
from .utils_io import ensure_outdir, savefig, rolling_groups_to_periods, periods_to_df, dedup_overlapping_periods

def rolling_mad_scores(s: pd.Series, win: int) -> pd.Series:
    med = s.rolling(win, min_periods=max(1, win//2)).median()
    mad = (s - med).abs().rolling(win, min_periods=max(1, win//2)).median()
    mad = mad.replace(0, np.nan)
    z = (s - med) / mad
    return z.abs()

def merge_anomaly_masks(mask_a: pd.Series, mask_b: pd.Series) -> pd.Series:
    if mask_a is None: return mask_b
    if mask_b is None: return mask_a
    return mask_a | mask_b

def contextual_anomalies(df: pd.DataFrame, state_labels: pd.Series) -> pd.DataFrame:
    """
    Train an IsolationForest per state on rolling features, plus rolling‑MAD on target.
    """
    cols = [c for c in CFG.SENSOR_COLS if c in df.columns]
    if not cols:
        cols = [CFG.TARGET_COL] if CFG.TARGET_COL in df.columns else list(df.columns)

    # MAD fallback on target
    step = int(CFG.RESAMPLE_RULE.replace("min",""))
    win = max(3, CFG.FEAT_WINDOWS_MIN[0] // step)
    mad_scores = rolling_mad_scores(df[CFG.TARGET_COL].astype(float), win) if CFG.TARGET_COL in df.columns else pd.Series(index=df.index, dtype=float)
    mad_mask = mad_scores > CFG.MAD_Z_K

    # Per‑state IsolationForest
    iforest_mask = pd.Series(False, index=df.index)
    iforest_score = pd.Series(np.nan, index=df.index, dtype=float)

    for st in sorted(set(state_labels.dropna().astype(int))):
        idx = state_labels[state_labels==st].index
        if len(idx) < 200:  # need enough samples
            continue
        feats = build_rolling_features(df.loc[idx, cols], cols)
        if len(feats) < 200:
            continue
        X = feats.values
        clf = IsolationForest(
            n_estimators=CFG.ISO_N_ESTIMATORS,
            max_samples=CFG.ISO_MAX_SAMPLES,
            contamination=CFG.ISO_CONTAM,
            random_state=CFG.RANDOM_SEED,
        )
        clf.fit(X)
        preds = clf.predict(X)  # -1 anomalous, 1 normal
        scores = -clf.decision_function(X)  # higher → more anomalous
        iforest_mask.loc[feats.index] = (preds == -1)
        iforest_score.loc[feats.index] = scores

    merged_mask = merge_anomaly_masks(iforest_mask, mad_mask)

    # Periodize & annotate
    periods = rolling_groups_to_periods(merged_mask.fillna(False))
    dfp = periods_to_df(periods)
    rows = []
    for _, r in dfp.iterrows():
        seg = slice(r["start"], r["end"])
        st = state_labels.loc[seg].mode(dropna=True)
        state = int(st.iloc[0]) if len(st) else -1
        # method aggregation
        m_if = bool(iforest_mask.loc[seg].any())
        m_mad = bool(mad_mask.loc[seg].any())
        method = "Both" if (m_if and m_mad) else ("IForest" if m_if else "MAD")
        score = np.nanmean(iforest_score.loc[seg].values)
        rows.append({
            "start": r["start"], "end": r["end"],
            "duration_min": r["duration_min"],
            "state": state, "method": method, "score": score
        })
    out = pd.DataFrame(rows)
    return out

def run_anomaly_detection(df: pd.DataFrame, outdir: str, state_labels: pd.Series):
    ensure_outdir(outdir)
    anomalies = contextual_anomalies(df, state_labels)
    anomalies = dedup_overlapping_periods(anomalies) if not anomalies.empty else anomalies

    csv_path = os.path.join(outdir, "anomalous_periods.csv")
    anomalies.to_csv(csv_path, index=False)
    logging.info("Saved anomalies: %s", csv_path)

    # Plot anomalies over target
    if CFG.TARGET_COL in df.columns and len(df)>0:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df.index, df[CFG.TARGET_COL], linewidth=0.8, label=CFG.TARGET_COL)
        for _, row in anomalies.iterrows():
            ax.axvspan(row["start"], row["end"], color="orange", alpha=0.2)
        ax.set_title("Anomalies (highlighted)")
        ax.set_xlabel("Time"); ax.set_ylabel("Value"); ax.legend()
        savefig(os.path.join(outdir, "anomalies_plot.png"), fig); plt.close(fig)

    return anomalies
