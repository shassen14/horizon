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
        if df.is_empty():
            return df
        df = df.sort("time")

        # Gather all expressions
        trend = self._get_trend_expressions()
        momentum = self._get_momentum_expressions()
        volatility = self._get_volatility_expressions()
        volume = self._get_volume_expressions()
        stats = self._get_statistical_expressions()
        calendar = self._get_calendar_expressions()
        structure = self._get_structural_expressions()

        # Filter out None/Empty lists
        all_exprs = [
            e
            for e in (
                trend + momentum + volatility + volume + stats + calendar + structure
            )
            if e is not None
        ]

        if all_exprs:
            df = df.with_columns(all_exprs)

        # 3. Calculate Relative Features (requires Join, must be done after)
        if benchmark_df is not None:
            df = self._add_relative_features(df, benchmark_df)

        return df

    def _get_trend_expressions(self) -> List[pl.Expr]:
        cfg = self.settings
        exprs = []

        # SMA
        for p in cfg.sma_periods:
            exprs.append(pl.col("close").rolling_mean(window_size=p).alias(f"sma_{p}"))

        # EMA
        for p in cfg.ema_periods:
            exprs.append(
                pl.col("close").ewm_mean(span=p, adjust=False).alias(f"ema_{p}")
            )

        # MACD
        macd_fast = pl.col("close").ewm_mean(span=cfg.macd_fast_period, adjust=False)
        macd_slow = pl.col("close").ewm_mean(span=cfg.macd_slow_period, adjust=False)
        macd_line = macd_fast - macd_slow
        macd_signal = macd_line.ewm_mean(span=cfg.macd_signal_period, adjust=False)

        exprs.append(macd_line.alias("macd"))
        exprs.append(macd_signal.alias("macd_signal"))
        exprs.append((macd_line - macd_signal).alias("macd_hist"))

        # ADX (Directional Movement)
        if self.talib_available:
            exprs.append(
                pl.map_batches(
                    [pl.col("high"), pl.col("low"), pl.col("close")],
                    lambda s: ta.ADX(
                        s[0].to_numpy(zero_copy_only=False),
                        s[1].to_numpy(zero_copy_only=False),
                        s[2].to_numpy(zero_copy_only=False),
                        timeperiod=cfg.adx_period,
                    ),
                    return_dtype=pl.Float64,  # Explicit Type
                ).alias(f"adx_{cfg.adx_period}")
            )

        return exprs

    def _get_momentum_expressions(self) -> List[pl.Expr]:
        cfg = self.settings
        exprs = []

        # 1. RSI
        if self.talib_available:
            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s: ta.RSI(
                        s.to_numpy(zero_copy_only=False), timeperiod=cfg.rsi_period
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"rsi_{cfg.rsi_period}")
            )

        # 2. Returns (Rate of Change)
        # We append each expression individually to the main list
        for p in cfg.roc_periods:
            exprs.append(pl.col("close").pct_change(n=p).alias(f"return_{p}"))

        # The result is a single flat list: [expr, expr, expr...]
        # NOT: [expr, [expr, expr]]
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
        atr_pct_expr = ((atr_expr / pl.col("close"))).alias(f"atr_{cfg.atr_period}_pct")
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
        - OBV (Requires TA-Lib)
        - MFI (Requires TA-Lib)
        """
        exprs = []

        # --- 1. Basic Volume Features (No TA-Lib needed) ---

        # Average Daily Volume (20 days)
        adv_20 = pl.col("volume").rolling_mean(window_size=20).alias("volume_adv_20")
        exprs.append(adv_20)

        # Relative Volume (Current Vol / ADV)
        rvol = (
            pl.when(adv_20 > 0).then(pl.col("volume") / adv_20).otherwise(None)
        ).alias("relative_volume")
        exprs.append(rvol)

        # --- 2. Advanced Volume Features (TA-Lib required) ---
        if self.talib_available:

            # On-Balance Volume (OBV)
            # Volume must be cast to float for TA-Lib
            obv_expr = pl.map_batches(
                [pl.col("close"), pl.col("volume")],
                lambda s: ta.OBV(
                    s[0].to_numpy(zero_copy_only=False),
                    s[1].to_numpy(zero_copy_only=False).astype(float),
                ),
                return_dtype=pl.Float64,
            ).alias("obv")
            exprs.append(obv_expr)

            # Money Flow Index (MFI)
            # MFI requires High, Low, Close, Volume
            mfi_expr = pl.map_batches(
                [pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")],
                lambda s: ta.MFI(
                    s[0].to_numpy(zero_copy_only=False),
                    s[1].to_numpy(zero_copy_only=False),
                    s[2].to_numpy(zero_copy_only=False),
                    s[3].to_numpy(zero_copy_only=False).astype(float),
                    timeperiod=self.settings.mfi_period,
                ),
                return_dtype=pl.Float64,
            ).alias(f"mfi_{self.settings.mfi_period}")
            exprs.append(mfi_expr)

        return exprs

    def _get_statistical_expressions(self) -> List[pl.Expr]:
        """
        Returns statistical features:
        - Rolling Skew (20)
        - Rolling Kurtosis (20)
        - Z-Score of Close (Distance from Mean / Std)
        """
        exprs = []

        # Polars has native rolling_skew (approx) and rolling_std
        # Window 20 (approx 1 month) and 60 (approx 1 quarter)

        for w in [20, 60]:
            # Skewness (Risk of crash)
            exprs.append(pl.col("close").rolling_skew(window_size=w).alias(f"skew_{w}"))

            # Z-Score: (Price - SMA) / StdDev
            # This is CRITICAL for ML. It tells the model "How many sigmas away is price?"
            sma = pl.col("close").rolling_mean(window_size=w)
            std = pl.col("close").rolling_std(window_size=w)
            z_score = ((pl.col("close") - sma) / std).alias(f"zscore_{w}")
            exprs.append(z_score)

        return exprs

    def _get_calendar_expressions(self) -> List[pl.Expr]:
        """Returns cyclic calendar features."""
        return [
            pl.col("time").dt.weekday().cast(pl.Float64).alias("day_of_week"),
            pl.col("time").dt.month().cast(pl.Float64).alias("month_of_year"),
            pl.col("time").dt.day().cast(pl.Float64).alias("day_of_month"),
            pl.col("time").dt.quarter().cast(pl.Float64).alias("quarter"),
        ]

    def _get_structural_expressions(self) -> List[pl.Expr]:
        """
        Returns Price Structure features (Distance from High/Low).
        Dynamic based on config.structural_periods.
        """
        exprs = []

        for p in self.settings.structural_periods:  # e.g. [252]
            # 1. Rolling High
            rolling_high = pl.col("high").rolling_max(window_size=p)

            # Calculate % Distance: (Close - High) / High
            # Result is negative (e.g., -0.05 means 5% below high)
            high_expr = ((pl.col("close") - rolling_high) / rolling_high).alias(
                f"high_{p}_pct"
            )

            exprs.append(high_expr)

            # 2. Rolling Low
            rolling_low = pl.col("low").rolling_min(window_size=p)

            # Calculate % Distance: (Close - Low) / Low
            # Result is positive (e.g., 0.10 means 10% above low)
            low_expr = ((pl.col("close") - rolling_low) / rolling_low).alias(
                f"low_{p}_pct"
            )

            exprs.append(low_expr)

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
