import polars as pl
from typing import List, Optional

# We wrap the import to handle environments where TA-Lib C-library isn't installed
try:
    import talib as ta

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: TA-Lib not found. Technical indicators will be skipped.")

from packages.quant_lib.config import Settings


class FeatureFactory:
    def __init__(self, settings: Settings):
        self.settings = settings.features
        self.talib_available = TALIB_AVAILABLE

    def generate_all(
        self, df: pl.DataFrame, benchmark_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Main entry point. Generates all configured features for a single asset dataframe.
        """
        if df.is_empty():
            return df

        # Ensure data is sorted by time for window functions
        df = df.sort("time")

        # 1. Gather all feature expressions
        trend_exprs = self._get_trend_expressions()
        momentum_exprs = self._get_momentum_expressions()
        volatility_exprs = self._get_volatility_expressions()
        volume_exprs = self._get_volume_expressions()

        # 2. Apply them in a single optimized pass
        # Note: We filter out None values in case TA-Lib is missing
        all_exprs = [
            e
            for e in (trend_exprs + momentum_exprs + volatility_exprs + volume_exprs)
            if e is not None
        ]

        if all_exprs:
            df = df.with_columns(all_exprs)

        # 3. Calculate Relative Features (requires Join, must be done after)
        if benchmark_df is not None:
            df = self._add_relative_features(df, benchmark_df)

        return df

    def _get_trend_expressions(self) -> List[pl.Expr]:
        """
        Returns expressions for:
        - SMA (20, 50, 200)
        - EMA (12, 20, 26, 50)
        - MACD (Line, Signal, Hist)
        """
        cfg = self.settings
        exprs = []

        # Simple Moving Averages
        for p in cfg.sma_periods:  # [20, 50, 200]
            exprs.append(pl.col("close").rolling_mean(window_size=p).alias(f"sma_{p}"))

        # Exponential Moving Averages
        for p in cfg.ema_periods:  # [12, 20, 26, 50]
            exprs.append(
                pl.col("close").ewm_mean(span=p, adjust=False).alias(f"ema_{p}")
            )

        # MACD
        # Uses standard settings (12, 26, 9) usually defined in config
        macd_fast = pl.col("close").ewm_mean(span=cfg.macd_fast_period, adjust=False)
        macd_slow = pl.col("close").ewm_mean(span=cfg.macd_slow_period, adjust=False)
        macd_line = macd_fast - macd_slow
        macd_signal = macd_line.ewm_mean(span=cfg.macd_signal_period, adjust=False)

        exprs.append(macd_line.alias("macd"))
        exprs.append(macd_signal.alias("macd_signal"))
        exprs.append((macd_line - macd_signal).alias("macd_hist"))

        return exprs

    def _get_momentum_expressions(self) -> List[pl.Expr]:
        """
        Returns expressions for:
        - RSI (14)
        - Returns (1d, 5d, 21d, 63d, 126d, 252d)
        """
        cfg = self.settings
        exprs = []

        # RSI (Requires TA-Lib)
        if self.talib_available:
            rsi_expr = (
                pl.col("close")
                .map_batches(
                    lambda s: ta.RSI(s.to_numpy(), timeperiod=cfg.rsi_period),
                    return_dtype=pl.Float64,
                )
                .alias(f"rsi_{cfg.rsi_period}")
            )
            exprs.append(rsi_expr)

        # Dynamic generation based on config list
        return_exprs = [
            pl.col("close").pct_change(n=p).alias(f"return_{p}")
            for p in cfg.roc_periods
        ]

        exprs.append(return_exprs)

        return exprs

    def _get_volatility_expressions(self) -> List[pl.Expr]:
        if not self.talib_available:
            return []

        cfg = self.settings
        exprs = []

        # --- ATR (No changes needed here) ---
        atr_expr = pl.map_batches(
            [pl.col("high"), pl.col("low"), pl.col("close")],
            lambda s: ta.ATR(
                s[0].to_numpy(),
                s[1].to_numpy(),
                s[2].to_numpy(),
                timeperiod=cfg.atr_period,
            ),
            return_dtype=pl.Float64,
        ).alias(f"atr_{cfg.atr_period}")

        exprs.append(atr_expr)
        atr_pct_expr = ((atr_expr / pl.col("close")) * 100).alias(
            f"atr_{cfg.atr_period}_pct"
        )
        exprs.append(atr_pct_expr)

        # --- Bollinger Bands (THE FIX) ---
        bb_struct = (
            pl.col("close")
            .map_batches(
                lambda s: pl.Series(  # <-- Wrap the output in pl.Series
                    [
                        {  # We create a list of dictionaries, one for each row
                            "upper": upper,
                            "middle": middle,
                            "lower": lower,
                        }
                        for upper, middle, lower in zip(
                            *ta.BBANDS(
                                s.to_numpy(),
                                timeperiod=cfg.bollinger_period,
                                nbdevup=cfg.bollinger_std_dev,
                                nbdevdn=cfg.bollinger_std_dev,
                            )
                        )
                    ]
                ),
                return_dtype=pl.Struct(
                    [
                        pl.Field("upper", pl.Float64),
                        pl.Field("middle", pl.Float64),
                        pl.Field("lower", pl.Float64),
                    ]
                ),
            )
            .alias("bb_struct")
        )

        # Unpacking logic remains the same
        bb_upper = bb_struct.struct.field("upper").alias(
            f"bb_upper_{cfg.bollinger_period}"
        )
        bb_middle = bb_struct.struct.field("middle").alias(
            f"bb_middle_{cfg.bollinger_period}"
        )
        bb_lower = bb_struct.struct.field("lower").alias(
            f"bb_lower_{cfg.bollinger_period}"
        )

        exprs.extend([bb_upper, bb_middle, bb_lower])

        bb_width = ((bb_upper - bb_lower) / bb_middle).alias(
            f"bb_width_{cfg.bollinger_period}"
        )
        exprs.append(bb_width)

        return exprs

    def _get_volume_expressions(self) -> List[pl.Expr]:
        """
        Returns expressions for:
        - ADV 20 (Average Daily Volume)
        - Relative Volume
        """
        exprs = []

        obv_expr = pl.map_batches(
            [pl.col("close"), pl.col("volume")],
            lambda s: ta.OBV(s[0].to_numpy(), s[1].to_numpy().astype(float)),
            return_dtype=pl.Float64,
        ).alias("obv")
        exprs.append(obv_expr)

        # Average Daily Volume (20 days)
        adv_20 = pl.col("volume").rolling_mean(window_size=20).alias("volume_adv_20")
        exprs.append(adv_20)

        # Relative Volume (Current Vol / ADV)
        # Use pl.when to avoid division by zero
        rvol = (
            pl.when(adv_20 > 0).then(pl.col("volume") / adv_20).otherwise(None)
        ).alias("relative_volume")
        exprs.append(rvol)

        return exprs

    def _add_relative_features(
        self, df: pl.DataFrame, benchmark_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculates Relative Strength vs a Benchmark (e.g. SPY).
        Requires joining, so it runs after the main expression pass.
        """
        # Validations
        if "time" not in benchmark_df.columns or "close" not in benchmark_df.columns:
            return df

        # 1. Prepare Benchmark
        bench = benchmark_df.select(
            [pl.col("time"), pl.col("close").alias("bench_close")]
        )

        # 2. Join
        df_merged = df.join(bench, on="time", how="left")

        # 3. Handle missing benchmark days (forward fill)
        df_merged = df_merged.with_columns(pl.col("bench_close").forward_fill())

        # 4. Calculate Ratio: (Stock / Benchmark)
        # We calculate the ratio, then normalize it by the FIRST available ratio
        # so all RS lines start at 1.0

        # Get the first valid ratio
        try:
            first_ratio = df_merged.select(
                (pl.col("close") / pl.col("bench_close")).drop_nulls().first()
            ).item()
        except Exception:
            first_ratio = None

        if first_ratio:
            rs_expr = ((pl.col("close") / pl.col("bench_close")) / first_ratio).alias(
                "rs_normalized"
            )

            df_merged = df_merged.with_columns(rs_expr)

        # Clean up temporary column
        df_merged = df_merged.drop("bench_close")

        return df_merged
