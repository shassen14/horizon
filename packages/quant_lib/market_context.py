import polars as pl
from typing import Dict


class MarketContextCalculator:
    """
    A collection of stateless methods for calculating market-level indicators.
    This class is pure logic and has no I/O dependencies.
    """

    def calculate_all_from_sources(
        self, asset_data: Dict[str, pl.DataFrame], breadth_features: pl.DataFrame | None
    ) -> pl.DataFrame:
        """
        Orchestrates the calculation and joining of all context features
        from pre-fetched data sources.
        """
        # Calculate each feature family from the provided data
        vix_context = self.calculate_vix_features(asset_data.get("VIX"))
        trend_context = self.calculate_trend_features(asset_data.get("SPY"))
        credit_context = self.calculate_credit_features(
            asset_data.get("HYG"), asset_data.get("IEF")
        )
        rates_context = self.calculate_rates_features(asset_data.get("TLT"))

        # Use breadth_features as the base DataFrame, as it's the most comprehensive
        final_df = breadth_features
        if final_df is None or final_df.is_empty():
            # Fallback to another source if breadth is missing, e.g., SPY timeline
            base_df = asset_data.get("SPY")
            if base_df is None or base_df.is_empty():
                return pl.DataFrame()  # Cannot proceed without a timeline
            final_df = base_df.select("time")

        # Sequentially join all other non-empty feature sets
        for feature_set in [vix_context, trend_context, credit_context, rates_context]:
            if feature_set is not None and not feature_set.is_empty():
                final_df = final_df.join(feature_set, on="time", how="left")

        # Sort and fill missing values that may arise from joins
        return final_df.sort("time").fill_null(strategy="forward")

    def calculate_asset_only_context(
        self, asset_data: Dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        vix = self.calculate_vix_features(asset_data.get("VIX"))
        trend = self.calculate_trend_features(asset_data.get("SPY"))
        credit = self.calculate_credit_features(
            asset_data.get("HYG"), asset_data.get("IEF")
        )
        rates = self.calculate_rates_features(asset_data.get("TLT"))

        # Collect all non-None dataframes
        valid_dfs = [
            df
            for df in [vix, trend, credit, rates]
            if df is not None and not df.is_empty()
        ]

        if not valid_dfs:
            return pl.DataFrame()

        # Start with the first one
        base = valid_dfs[0]

        # Join the rest
        for fs in valid_dfs[1:]:
            base = base.join(fs, on="time", how="left")

        return base.sort("time").fill_null(strategy="forward")

    def calculate_asset_only_context(
        self, asset_data: Dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Calculates only the features derived from single assets (VIX, SPY, Rates, Credit).
        Does NOT include Breadth.
        """
        vix = self.calculate_vix_features(asset_data.get("VIX"))
        trend = self.calculate_trend_features(asset_data.get("SPY"))
        credit = self.calculate_credit_features(
            asset_data.get("HYG"), asset_data.get("IEF")
        )
        rates = self.calculate_rates_features(asset_data.get("TLT"))

        # Join them all
        base = vix if vix is not None else trend
        for fs in [trend, credit, rates]:
            if fs is not None and base is not None:
                base = base.join(fs, on="time", how="left")
            elif base is None:
                base = fs

        if base is not None:
            return base.sort("time").forward_fill()
        return pl.DataFrame()

    def calculate_vix_features(self, vix_df: pl.DataFrame) -> pl.DataFrame | None:
        if vix_df is None or vix_df.is_empty():
            return None

        return vix_df.select(
            [
                pl.col("time"),
                pl.col("close").alias("vix_close"),
                pl.col("close").pct_change(1).alias("vix_pct_change_1d"),
            ]
        )

    def calculate_breadth_features(
        self, all_features_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        if all_features_df is None or all_features_df.is_empty():
            return None

        # Group by time and calculate the mean of boolean conditions (which gives the percentage)
        return all_features_df.group_by("time").agg(
            [
                (pl.col("close") > pl.col("sma_20"))
                .mean()
                .alias("breadth_pct_above_sma20"),
                (pl.col("close") > pl.col("sma_50"))
                .mean()
                .alias("breadth_pct_above_sma50"),
                (pl.col("close") > pl.col("sma_200"))
                .mean()
                .alias("breadth_pct_above_sma200"),
            ]
        )

    def calculate_trend_features(self, spy_df: pl.DataFrame) -> pl.DataFrame | None:
        if spy_df is None or spy_df.is_empty():
            return None

        # These features are already pre-calculated in features_daily for SPY
        return spy_df.select(
            [
                pl.col("time"),
                pl.col("rsi_14").alias("spy_rsi_14"),
                pl.col("adx_14").alias("spy_adx_14"),
            ]
        )

    def calculate_credit_features(
        self, hyg_df: pl.DataFrame, ief_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        if hyg_df is None or ief_df is None or hyg_df.is_empty() or ief_df.is_empty():
            return None

        hyg_price = hyg_df.select(["time", "close"]).rename({"close": "hyg"})
        ief_price = ief_df.select(["time", "close"]).rename({"close": "ief"})

        spread_df = hyg_price.join(ief_price, on="time", how="inner")

        return spread_df.with_columns(
            (pl.col("hyg") / pl.col("ief"))
            .pct_change(5)
            .alias("credit_spread_pct_change_5d")
        ).select(["time", "credit_spread_pct_change_5d"])

    def calculate_rates_features(self, tlt_df: pl.DataFrame) -> pl.DataFrame | None:
        if tlt_df is None or tlt_df.is_empty():
            return None

        return tlt_df.with_columns(
            pl.col("close").pct_change(21).alias("tlt_pct_change_21d")
        ).select(["time", "tlt_pct_change_21d"])
