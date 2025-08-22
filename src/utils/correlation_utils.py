import pandas as pd
import numpy as np
from typing import List, Optional

def prepare_corr_frame(df: pd.DataFrame, leakage_cols: Optional[List[str]] = None) -> pd.DataFrame:
    leakage_cols = leakage_cols or []
    num = (df.select_dtypes(include=[np.number])
             .replace([np.inf, -np.inf], np.nan)
             .dropna(axis=1, how='all'))
    drop = set(leakage_cols) & set(num.columns)
    if drop:
        num = num.drop(columns=list(drop))
    std = num.std(ddof=0)
    return num.loc[:, std > 0]