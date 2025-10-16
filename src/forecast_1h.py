import os
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA

from .config import CFG
from .utils_io import ensure_outdir, savefig

def has_daily_seasonality(s: pd.Series) -> bool:
    # Simple variance ratio around daily lag
    lag = CFG.DAILY_STEPS
    if len(s) <= lag+1:
        return False
    x = s.dropna().values
    if len(x) <= lag+1:
        return False
    corr = np.corrcoef(x[:-lag], x[lag:])[0,1]
    return bool(corr >= 0.3)  # sensible threshold

def make_supervised(y: pd.Series, n_lags: int, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame({"y": y})
    for i in range(1, n_lags+1):
        df[f"lag_{i}"] = df["y"].shift(i)
    # forecast horizon offset
    df["y+H"] = df["y"].shift(-horizon)
    df = df.dropna()
    X = df[[c for c in df.columns if c.startswith("lag_")]]
    return X, df["y+H"]

def naive_forecast(y: pd.Series, horizon: int) -> pd.Series:
    # persistence (last observed value)
    return y.shift(0)  # will be aligned by backtest logic

def seasonal_naive_forecast(y: pd.Series, horizon: int, season: int) -> pd.Series:
    return y.shift(season)

def arima_best(y_train: pd.Series) -> Tuple[Tuple[int,int,int], ARIMA]:
    best_aic = np.inf; best_order = None; best_model = None
    for p in CFG.ARIMA_P:
        for d in CFG.ARIMA_D:
            for q in CFG.ARIMA_Q:
                try:
                    model = ARIMA(y_train, order=(p,d,q))
                    fit = model.fit(method_kwargs={"warn_convergence": False})
                    if fit.aic < best_aic:
                        best_aic = fit.aic; best_order=(p,d,q); best_model=fit
                except Exception:
                    continue
    if best_model is None:
        # fallback
        best_order=(1,1,1)
        best_model = ARIMA(y_train, order=best_order).fit()
    logging.info("ARIMA best order=%s AIC=%.1f", best_order, best_aic)
    return best_order, best_model

def walk_forward_backtest(y: pd.Series, outdir: str):
    ensure_outdir(outdir)
    H = CFG.HORIZON_STEPS
    step = pd.to_timedelta(CFG.RESAMPLE_RULE)

    # define rolling windows over last BACKTEST_DAYS with min train
    last_day = y.index.max().normalize()
    start_test = last_day - pd.Timedelta(days=CFG.BACKTEST_DAYS)
    y_sub = y[y.index >= (start_test - pd.Timedelta(days=CFG.MIN_TRAIN_DAYS))]

    timestamps = []
    rows = []
    forecasts_rows = []

    # Precompute seasonal availability
    use_seasonal = has_daily_seasonality(y_sub)

    # Rolling split every H steps
    idx = y_sub.index
    split_points = list(range(CFG.DAILY_STEPS, len(idx)-H, H))
    for sp in split_points:
        t0 = idx[sp]  # forecast origin time
        train = y_sub.loc[:t0]
        test_idx = idx[sp+H-1]  # approximate horizon end

        # Naive
        yhat_naive = pd.Series(train.iloc[-1], index=[idx[sp+H-1]])
        # Seasonal-naive
        if use_seasonal:
            if sp - CFG.DAILY_STEPS >= 0:
                yhat_seas = pd.Series(y_sub.iloc[sp - CFG.DAILY_STEPS + H -1], index=[idx[sp+H-1]])
            else:
                yhat_seas = pd.Series(np.nan, index=[idx[sp+H-1]])
        else:
            yhat_seas = pd.Series(np.nan, index=[idx[sp+H-1]])

        # ARIMA (fit on train)
        try:
            order, ar_fit = arima_best(train.dropna())
            ar_fore = ar_fit.forecast(steps=H)
            yhat_arima = pd.Series(ar_fore.iloc[-1], index=[idx[sp+H-1]])
        except Exception as e:
            logging.warning("ARIMA failed at %s: %s", t0, e)
            yhat_arima = pd.Series(np.nan, index=[idx[sp+H-1]])

        # GradientBoosting on lags
        X, yy = make_supervised(y_sub, CFG.GB_LAGS, H)
        Xtr = X[X.index <= t0]; ytr = yy[X.index <= t0]
        Xte = X[X.index > t0].iloc[:1]  # next target point
        if len(Xtr) >= 200 and len(Xte)==1:
            gbr = GradientBoostingRegressor(random_state=CFG.RANDOM_SEED)
            gbr.fit(Xtr.values, ytr.values)
            yhat_gb = pd.Series(float(gbr.predict(Xte.values)[0]), index=[Xte.index[0]])
        else:
            yhat_gb = pd.Series(np.nan, index=[idx[sp+H-1]])

        # True value to compare at horizon
        y_true = y_sub.loc[yhat_naive.index[0]] if yhat_naive.index[0] in y_sub.index else np.nan

        # Accumulate forecasts.csv rows
        forecasts_rows += [
            {"timestamp": yhat_naive.index[0], "y_true": y_true, "y_pred": float(yhat_naive.iloc[0]), "model": "naive"},
            {"timestamp": yhat_seas.index[0],  "y_true": y_true, "y_pred": float(yhat_seas.iloc[0]) if not np.isnan(yhat_seas.iloc[0]) else np.nan, "model": "seasonal_naive" if use_seasonal else "seasonal_naive_disabled"},
            {"timestamp": yhat_arima.index[0], "y_true": y_true, "y_pred": float(yhat_arima.iloc[0]) if not np.isnan(yhat_arima.iloc[0]) else np.nan, "model": "arima"},
            {"timestamp": yhat_gb.index[0],    "y_true": y_true, "y_pred": float(yhat_gb.iloc[0]) if not np.isnan(yhat_gb.iloc[0]) else np.nan, "model": "gbr_lags"},
        ]

    # Compile and score
    fdf = pd.DataFrame(forecasts_rows).dropna(subset=["y_true","y_pred"])
    metrics = []
    for model, g in fdf.groupby("model"):
        mae = mean_absolute_error(g["y_true"], g["y_pred"])
        rmse = mean_squared_error(g["y_true"], g["y_pred"], squared=False)
        metrics.append({"model": model, "MAE": mae, "RMSE": rmse, "n": len(g)})
        logging.info("Backtest %s: MAE=%.3f RMSE=%.3f n=%d", model, mae, rmse, len(g))

    return fdf, pd.DataFrame(metrics).sort_values("RMSE")

def run_forecast(df: pd.DataFrame, outdir: str):
    ensure_outdir(outdir)
    if CFG.TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{CFG.TARGET_COL}' not in data. Set in src/config.py or pass via preprocessing.")

    y = df[CFG.TARGET_COL].astype(float).dropna()
    forecasts, metrics = walk_forward_backtest(y, outdir)

    fcsv = os.path.join(outdir, "forecasts.csv")
    mcsv = os.path.join(outdir, "backtest_metrics.csv")
    forecasts.to_csv(fcsv, index=False); metrics.to_csv(mcsv, index=False)
    logging.info("Saved forecasts: %s", fcsv)
    logging.info("Saved metrics: %s", mcsv)

    # Plot last 7 days with predictions from best model
    if not forecasts.empty:
        best_model = metrics.iloc[0]["model"]
        sub = forecasts[forecasts["model"]==best_model].tail(288*7)  # ~7 days
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(sub["timestamp"], sub["y_true"], label="y_true", linewidth=0.8)
        ax.plot(sub["timestamp"], sub["y_pred"], label=f"y_pred ({best_model})", linewidth=0.8)
        ax.set_title("1h‑ahead forecast (best model)")
        ax.set_xlabel("Time"); ax.set_ylabel("Value"); ax.legend()
        savefig(os.path.join(outdir, "forecast_plot.png"), fig); plt.close(fig)

    return forecasts, metrics
