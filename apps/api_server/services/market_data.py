# apps/api_server/services/market_data.py

from datetime import datetime, timedelta
from typing import List, Optional
import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api_server.schemas.market import (
    HistoryDataPoint,
    FeatureSet,
    TrendFeatures,
    MomentumFeatures,
    VolatilityFeatures,
    VolumeFeatures,
)
from apps.api_server.schemas.enums import MarketInterval
from apps.api_server.core.utils import interval_to_minutes
from packages.database.models import (
    Asset,
    MarketDataDaily,
    MarketData5Min,
    FeaturesDaily,
)
from packages.quant_lib.features import FeatureFactory
from packages.quant_lib.config import settings


class MarketDataService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.feature_factory = FeatureFactory(settings)

    async def get_history(
        self,
        symbol: str,
        interval: MarketInterval,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int,
    ) -> List[HistoryDataPoint]:
        """
        Main entry point. Decides strategy based on requested interval.
        """
        # 1. Resolve Asset ID (Fast)
        asset_id = await self.session.scalar(
            select(Asset.id).where(Asset.symbol == symbol.upper())
        )
        if not asset_id:
            return []

        # 2. Strategy Selector
        if interval == MarketInterval.DAY_1:
            return await self._get_daily_strategy(asset_id, start_date, end_date, limit)
        elif interval == MarketInterval.WEEK_1:
            return await self._get_derived_strategy(
                asset_id, MarketDataDaily, interval, start_date, end_date, limit
            )
        else:
            # All intraday (1h, 4h, 15m) derives from 5Min
            return await self._get_derived_strategy(
                asset_id, MarketData5Min, interval, start_date, end_date, limit
            )

    async def _get_daily_strategy(
        self, asset_id: int, start: datetime, end: datetime, limit: int
    ):
        """
        FAST PATH: Query Daily tables directly. No resampling needed.
        """
        stmt = (
            select(MarketDataDaily.__table__, FeaturesDaily.__table__)
            .join(
                FeaturesDaily.__table__,
                (FeaturesDaily.asset_id == MarketDataDaily.asset_id)
                & (FeaturesDaily.time == MarketDataDaily.time),
                isouter=True,
            )
            .where(MarketDataDaily.asset_id == asset_id)
        )

        # Apply Filters
        if start:
            stmt = stmt.where(MarketDataDaily.time >= start)
        if end:
            stmt = stmt.where(MarketDataDaily.time <= end)

        # Sorting
        # If fetching a range -> ASC. If fetching latest N -> DESC then reverse.
        is_range = start is not None
        if is_range:
            stmt = stmt.order_by(MarketDataDaily.time.asc())
        else:
            stmt = stmt.order_by(MarketDataDaily.time.desc())

        stmt = stmt.limit(limit)

        result = await self.session.execute(stmt)
        rows = result.mappings().all()

        if not is_range:
            rows = rows[::-1]

        return [self._map_row_to_point(row) for row in rows]

    async def _get_derived_strategy(
        self,
        asset_id: int,
        SourceModel,
        interval: MarketInterval,
        start: datetime,
        end: datetime,
        limit: int,
    ):
        """
        COMPLEX PATH: Fetch high-res data, Resample, Calculate Features.
        Optimized to use Time-Based Indexing when start_date is provided.
        """
        # Constants
        WARMUP_BARS = 300  # Enough for SMA-200 + stabilization

        # 1. Determine "Base" Resolution of the Source Table
        # (Assuming MarketDataDaily is 1440m, MarketData5Min is 5m)
        if SourceModel.__tablename__ == "market_data_daily":
            base_minutes = 1440
        else:
            # We could look this up dynamically or hardcode 5 for now
            base_minutes = 5

        # 2. Build the Base Query
        # We always want the newest data first for consistency in fetching logic,
        # but we will sort ASC later for Polars.
        stmt = (
            select(
                SourceModel.time,
                SourceModel.open,
                SourceModel.high,
                SourceModel.low,
                SourceModel.close,
                SourceModel.volume,
            )
            .where(SourceModel.asset_id == asset_id)
            .order_by(SourceModel.time.desc())
        )

        # 3. Dynamic Strategy Selection

        # --- STRATEGY A: TIME RANGE (The Fix) ---
        if start:
            # Calculate Warmup Delta (Time) instead of Rows
            # We need X bars * Y minutes/bar.
            # We multiply by 4 to account for weekends/market closures safely.
            warmup_minutes = WARMUP_BARS * base_minutes * 4
            query_start = start - timedelta(minutes=warmup_minutes)

            # Apply Time Filters
            stmt = stmt.where(SourceModel.time >= query_start)
            if end:
                stmt = stmt.where(SourceModel.time <= end)

            # We do NOT apply the 'limit' param here strictly for fetching,
            # because we want to fill the time range exactly.
            # But we apply a "Sanity Cap" to prevent fetching 10 years by accident.
            SANITY_LIMIT = 200_000
            stmt = stmt.limit(SANITY_LIMIT)

        # --- STRATEGY B: LATEST N ROWS (Fallback) ---
        else:
            # Calculate Rows needed based on Target Interval
            target_minutes = interval_to_minutes(interval)
            resample_factor = max(1, target_minutes // base_minutes)

            total_target_bars = limit + WARMUP_BARS
            raw_rows_needed = int(total_target_bars * resample_factor)

            # Safety Cap
            raw_rows_needed = min(raw_rows_needed, 100_000)

            stmt = stmt.limit(raw_rows_needed)
            if end:
                stmt = stmt.where(SourceModel.time <= end)

        # 4. Execute Query
        result = await self.session.execute(stmt)
        data = [dict(r) for r in result.mappings().all()]

        if not data:
            return []

        # 5. Resample with Polars (Same as before)
        df = pl.DataFrame(data).sort("time").set_sorted("time")

        # Helper to convert Enum to Polars Interval string (e.g. "1w")
        # Polars expects "1w", "1h", "15m" (lowercase)
        pl_interval = interval.value.lower()

        df_agg = df.group_by_dynamic("time", every=pl_interval).agg(
            [
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
            ]
        )

        # 6. Calculate Features (Same as before)
        try:
            df_features = self.feature_factory.generate_all(df_agg, benchmark_df=None)
        except Exception:
            df_features = df_agg

        # 7. Precise Trimming (Critical for Clean UI)
        if start:
            # If User asked for Start Date, cut off exactly at Start Date
            # (Removing the warmup data we fetched)
            df_features = df_features.filter(pl.col("time") >= start)

            # Optional: Apply limit after time filtering if desired
            if limit and len(df_features) > limit:
                df_features = df_features.head(limit)  # Take first N after start
        else:
            # If User asked for "Latest", take the last N
            rows_to_keep = min(len(df_features), limit)
            df_features = df_features.tail(rows_to_keep)

        return [self._map_row_to_point(row) for row in df_features.to_dicts()]

    def _map_row_to_point(self, row: dict) -> HistoryDataPoint:
        """Helper to map a dictionary (DB or Polars) to the Pydantic model."""

        # Helper to safely get nested dicts/keys
        def get(key):
            return row.get(key)

        return HistoryDataPoint(
            time=row["time"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            features=FeatureSet(
                trend=TrendFeatures(
                    sma_20=get("sma_20"),
                    sma_50=get("sma_50"),
                    sma_200=get("sma_200"),
                    ema_12=get("ema_12"),
                    ema_20=get("ema_20"),
                    ema_26=get("ema_26"),
                    ema_50=get("ema_50"),
                    macd=get("macd"),
                    macd_signal=get("macd_signal"),
                    macd_hist=get("macd_hist"),
                ),
                momentum=MomentumFeatures(
                    rsi_14=get("rsi_14"),
                    return_1=get("return_1"),
                    return_5=get("return_5"),
                    return_21=get("return_21"),
                    return_63=get("return_63"),
                ),
                volatility=VolatilityFeatures(
                    atr_14=get("atr_14"),
                    atr_14_pct=get("atr_14_pct"),
                    bb_upper_20=get("bb_upper_20"),
                    bb_middle_20=get("bb_middle_20"),
                    bb_lower_20=get("bb_lower_20"),
                    bb_width_20=get("bb_width_20"),
                ),
                volume=VolumeFeatures(
                    volume_adv_20=get("volume_adv_20"),
                    relative_volume=get("relative_volume"),
                ),
            ),
        )
