import numpy as np
import pandas as pd
from typing import List
from .base import BasePermutator


class OHLCPermutator(BasePermutator):
    """
    Implementation of Combinatorial Purged Cross-Validation permutation.

    1. Deconstructs Price into Log-Returns.
    2. Deconstructs Volume into Ratios.
    3. Deconstructs "Extra" columns (like Breadth) into arithmetic differences.
    4. Shuffles the deltas synchronously to preserve cross-correlations.
    5. Reconstructs a synthetic time series from a fixed starting point.
    """

    def __init__(self, diff_cols: List[str] = None):
        # Columns to be permuted via arithmetic difference instead of log/ratio
        self.diff_cols = diff_cols if diff_cols else []

    def permute(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        np.random.seed(seed)

        # Ensure data is sorted by time for correct diff/shift operations
        df_sorted = df.sort_values("time").reset_index(drop=True)
        n_bars = len(df_sorted)
        if n_bars < 2:
            return df_sorted

        # --- 1. DECONSTRUCTION ---
        # A. Log Prices
        opens = np.log(df_sorted["open"].to_numpy())
        closes = np.log(df_sorted["close"].to_numpy())
        highs = np.log(df_sorted["high"].to_numpy())
        lows = np.log(df_sorted["low"].to_numpy())

        # B. Volume (handle zeros to prevent errors)
        vols = np.where(
            df_sorted["volume"].to_numpy() == 0, 1, df_sorted["volume"].to_numpy()
        )

        # C. Difference Columns (e.g., Breadth)
        diff_data = {
            col: df_sorted[col].to_numpy()
            for col in self.diff_cols
            if col in df_sorted.columns
        }

        # --- 2. CALCULATE RELATIVES/DELTAS ---
        r_open = opens[1:] - closes[:-1]  # Gap
        r_high = highs[1:] - opens[1:]  # Intraday High
        r_low = lows[1:] - opens[1:]  # Intraday Low
        r_close = closes[1:] - opens[1:]  # Intraday Close
        r_vol = vols[1:] / vols[:-1]  # Volume Ratio
        r_diffs = {col: data[1:] - data[:-1] for col, data in diff_data.items()}

        # --- 3. SHUFFLE ---
        perm_n = n_bars - 1
        # Permutation for intraday structure
        perm1 = np.random.permutation(perm_n)
        # Permutation for interday (gap) structure
        perm2 = np.random.permutation(perm_n)

        r_high, r_low, r_close, r_vol = (
            r_high[perm1],
            r_low[perm1],
            r_close[perm1],
            r_vol[perm1],
        )
        for col in r_diffs:
            r_diffs[col] = r_diffs[col][perm1]
        r_open = r_open[perm2]

        # --- 4. RECONSTRUCTION ---
        new_opens, new_highs, new_lows, new_closes = (
            np.zeros(n_bars) for _ in range(4)
        )
        new_vols = np.zeros(n_bars)
        new_extras = {col: np.zeros(n_bars) for col in self.diff_cols}

        # Anchor the start bar
        new_opens[0], new_highs[0], new_lows[0], new_closes[0], new_vols[0] = (
            opens[0],
            highs[0],
            lows[0],
            closes[0],
            vols[0],
        )
        for col, data in diff_data.items():
            new_extras[col][0] = data[0]

        # Iteratively reconstruct the series
        for i in range(perm_n):
            # Price
            o = new_closes[i] + r_open[i]
            new_opens[i + 1], new_highs[i + 1], new_lows[i + 1], new_closes[i + 1] = (
                o,
                o + r_high[i],
                o + r_low[i],
                o + r_close[i],
            )
            # Volume
            new_vols[i + 1] = new_vols[i] * r_vol[i]
            # Extras
            for col in self.diff_cols:
                val = new_extras[col][i] + r_diffs[col][i]
                # Clip percentage-based features to a valid [0, 1] range
                if "pct_" in col:
                    val = np.clip(val, 0.0, 1.0)
                new_extras[col][i + 1] = val

        # --- 5. FINALIZE DATAFRAME ---
        syn_df = df_sorted.copy()
        syn_df["open"], syn_df["high"], syn_df["low"], syn_df["close"] = (
            np.exp(new_opens),
            np.exp(new_highs),
            np.exp(new_lows),
            np.exp(new_closes),
        )
        syn_df["volume"] = np.round(new_vols)  # Volume should be integer-like
        for col, data in new_extras.items():
            syn_df[col] = data

        return syn_df
