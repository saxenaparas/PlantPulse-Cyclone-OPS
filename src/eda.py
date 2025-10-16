import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import CFG
from .utils_io import summarize, mad_clip_series, savefig, ensure_outdir

def run_eda(df: pd.DataFrame, outdir: str):
    ensure_outdir(outdir)
    # Robust outlier handling: MAD-based winsorization per column (EDA summaries only, not fed into models directly)
    df_clipped = df.copy()
    for c in df_clipped.columns:
        df_clipped[c] = mad_clip_series(df_clipped[c], CFG.MAD_CLIP_K)

    # Summary stats
    summ = summarize(df_clipped)
    out_csv = os.path.join(outdir, "eda_summary.csv")
    summ.to_csv(out_csv)
    logging.info("Saved EDA summary: %s", out_csv)

    # Correlations
    corr = df_clipped.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr.values, aspect='auto')
    ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(os.path.join(outdir, "correlations.png"), fig); plt.close(fig)

    # Week and Year views (pick first complete week/year available)
    if len(df_clipped) > 0:
        freq = CFG.RESAMPLE_RULE
        step = pd.to_timedelta(freq)
        week = df_clipped.iloc[: (7*24*60)//int(freq.replace("min","")) ]
        year = df_clipped.iloc[: (365*24*60)//int(freq.replace("min","")) ]
        # Week plot
        figw, axw = plt.subplots(figsize=(10,4))
        axw.plot(week.index, week[CFG.TARGET_COL], label=CFG.TARGET_COL)
        axw.set_title("Week view: target"); axw.set_xlabel("Time"); axw.set_ylabel("Value"); axw.legend()
        savefig(os.path.join(outdir, "week_view.png"), figw); plt.close(figw)
        # Year plot
        figy, axy = plt.subplots(figsize=(12,4))
        axy.plot(year.index, year[CFG.TARGET_COL], label=CFG.TARGET_COL)
        axy.set_title("Year view: target"); axy.set_xlabel("Time"); axy.set_ylabel("Value"); axy.legend()
        savefig(os.path.join(outdir, "year_view.png"), figy); plt.close(figy)
