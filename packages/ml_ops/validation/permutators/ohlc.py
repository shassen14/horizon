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
        df_sorted = df.sort_values("time").reset_index(drop=True)
        n_bars = len(df_sorted)
        if n_bars < 2:
            return df_sorted

        # --- DECONSTRUCT PRICES & VOL ---
        opens = np.log(df_sorted["open"].to_numpy())
        closes = np.log(df_sorted["close"].to_numpy())
        highs = np.log(df_sorted["high"].to_numpy())
        lows = np.log(df_sorted["low"].to_numpy())
        vols = np.where(
            df_sorted["volume"].to_numpy() == 0, 1, df_sorted["volume"].to_numpy()
        )

        # --- CALCULATE PRICE/VOL RELATIVES ---
        r_open = opens[1:] - closes[:-1]
        r_high = highs[1:] - opens[1:]
        r_low = lows[1:] - opens[1:]
        r_close = closes[1:] - opens[1:]
        r_vol = vols[1:] / vols[:-1]

        # --- SHUFFLE PRICE/VOL ---
        perm_n = n_bars - 1
        perm1 = np.random.permutation(perm_n)
        perm2 = np.random.permutation(perm_n)

        r_high, r_low, r_close, r_vol = (
            r_high[perm1],
            r_low[perm1],
            r_close[perm1],
            r_vol[perm1],
        )
        r_open = r_open[perm2]

        # --- RECONSTRUCT PRICE/VOL ---
        new_opens, new_highs, new_lows, new_closes = (
            np.zeros(n_bars) for _ in range(4)
        )

        new_opens[0], new_highs[0], new_lows[0], new_closes[0] = (
            opens[0],
            highs[0],
            lows[0],
            closes[0],
        )
        for i in range(perm_n):
            o = new_closes[i] + r_open[i]
            new_opens[i + 1], new_highs[i + 1], new_lows[i + 1], new_closes[i + 1] = (
                o,
                o + r_high[i],
                o + r_low[i],
                o + r_close[i],
            )

        # --- HANDLE EXTRA COLUMNS (BREADTH) ---
        # **  Shuffle the values directly, not the differences. **
        syn_df = df_sorted.copy()
        cols_to_shuffle = self.diff_cols + ["volume"]

        for col in cols_to_shuffle:
            if col in syn_df.columns:
                original_values = syn_df[col].to_numpy()
                permuted_values = np.concatenate(
                    ([original_values[0]], np.random.permutation(original_values[1:]))
                )
                syn_df[col] = permuted_values

        # --- FINALIZE DATAFRAME ---
        syn_df["open"], syn_df["high"], syn_df["low"], syn_df["close"] = (
            np.exp(new_opens),
            np.exp(new_highs),
            np.exp(new_lows),
            np.exp(new_closes),
        )

        return syn_df
