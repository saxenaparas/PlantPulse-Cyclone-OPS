from typing import List
import numpy as np
import pandas as pd
from .config import CFG

def build_rolling_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Create rolling window features for each column:
    - mean, std over windows (minutes in CFG.FEAT_WINDOWS_MIN)
    - first difference (1 step) as short-term gradient
    Drops rows with NaNs created by rolling.
    """
    out = pd.DataFrame(index=df.index)
    step_min = int(CFG.RESAMPLE_RULE.replace("min",""))
    for c in cols:
        s = df[c].astype(float)
        out[f"{c}_diff1"] = s.diff(1)
        for wmin in CFG.FEAT_WINDOWS_MIN:
            w = max(1, wmin // step_min)
            out[f"{c}_mean_{wmin}m"] = s.rolling(w, min_periods=max(1, w//2)).mean()
            out[f"{c}_std_{wmin}m"]  = s.rolling(w, min_periods=max(1, w//2)).std()
    out = out.dropna()
    return out
