import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import CFG
from .features import build_rolling_features
from .utils_io import ensure_outdir, savefig

# Human‑readable mapping (example guidance; can be refined by domain SMEs)
STATE_LABELS = {
    0: "State_0",
    1: "State_1",
    2: "State_2",
    3: "State_3",
    4: "State_4",
    5: "State_5",
    6: "State_6",
    7: "State_7",
}
# Example interpretation in comments:
# - Higher inlet/outlet temps + moderate drafts → "High Load"
# - Low drafts + rising std → "Unstable/Transition"
# - etc. Mapping is data‑dependent; keep neutral labels here.

def best_kmeans(X: np.ndarray, kmin: int, kmax: int, seed: int):
    sil_scores, inertias = [], []
    best = (None, -np.inf, None)
    for k in range(kmin, kmax+1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(20000, len(X)), random_state=CFG.RANDOM_SEED) if k > 1 else -1.0
        sil_scores.append((k, sil))
        inertias.append((k, km.inertia_))
        if sil > best[1]:
            best = (km, sil, labels)
    return best, sil_scores, inertias

def run_clustering(df: pd.DataFrame, outdir: str, exclude_mask: pd.Series = None):
    ensure_outdir(outdir)

    # Select sensor columns present
    cols = [c for c in CFG.SENSOR_COLS if c in df.columns]
    if not cols:
        cols = [CFG.TARGET_COL] if CFG.TARGET_COL in df.columns else list(df.columns)

    # Exclude shutdowns if provided
    if exclude_mask is not None and len(exclude_mask)==len(df):
        df_work = df.loc[~exclude_mask].copy()
    else:
        df_work = df.copy()

    # Feature matrix
    feats = build_rolling_features(df_work, cols)
    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    # KMeans selection
    (km, best_sil, labels), sil_scores, inertias = best_kmeans(
        X, CFG.KMIN, CFG.KMAX, CFG.RANDOM_SEED
    )
    logging.info("Chosen K=%d with silhouette=%.3f", km.n_clusters, best_sil)

    # Attach labels back on the aligned index
    df_states = pd.DataFrame(index=feats.index)
    df_states["cluster"] = labels
    df_states["state_name"] = df_states["cluster"].map(STATE_LABELS).fillna("State")

    # Summaries per cluster on original sensor columns
    joined = df_work.loc[feats.index, cols].join(df_states["cluster"])
    summary = joined.groupby("cluster").agg(["mean","std","median","min","max"])
    summary.index.name = "cluster"
    summary_path = os.path.join(outdir, "clusters_summary.csv")
    summary.to_csv(summary_path)
    logging.info("Saved clusters summary: %s", summary_path)

    # Elbow and silhouette plots
    k_vals, sil_vals = zip(*sil_scores)
    _, inert_vals = zip(*inertias)
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(k_vals, inert_vals, marker="o"); ax1.set_xlabel("K"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow")
    savefig(os.path.join(outdir, "elbow.png"), fig1); plt.close(fig1)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(k_vals, sil_vals, marker="o"); ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette"); ax2.set_title("Silhouette vs K")
    savefig(os.path.join(outdir, "silhouette.png"), fig2); plt.close(fig2)

    # 2D scatter via PCA
    pca = PCA(n_components=CFG.PCA_COMPONENTS, random_state=CFG.RANDOM_SEED)
    X2 = pca.fit_transform(X)
    fig3, ax3 = plt.subplots(figsize=(6,5))
    sc = ax3.scatter(X2[:,0], X2[:,1], c=labels, s=8, alpha=0.6)
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_title("Cluster scatter (PCA)")
    savefig(os.path.join(outdir, "cluster_scatter.png"), fig3); plt.close(fig3)

    # Return labels reindexed to the full df timeline (NaN during excluded/shutdown periods)
    full_labels = pd.Series(np.nan, index=df.index, dtype=float)
    full_labels.loc[feats.index] = labels
    return df_states, full_labels

