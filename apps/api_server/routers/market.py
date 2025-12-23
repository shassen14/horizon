# apps/api_server/routers/market.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

# Absolute imports are now clean and predictable
from apps.api_server.core.limiter import limiter
from apps.api_server.dependencies.database import get_session
from apps.api_server.schemas.market import (
    HistoryDataPoint,
    FeatureSet,
    TrendFeatures,
    MomentumFeatures,
    VolatilityFeatures,
    VolumeFeatures,
    MarketSnapshot,
)
from packages.database.models import Asset, MarketDataDaily, FeaturesDaily

router = APIRouter(prefix="/public/market", tags=["Market Data"])


@router.get(
    "/history/{symbol}",
    response_model=List[HistoryDataPoint],
    operation_id="get_symbol_history",
)
@limiter.limit("100/minute")
async def get_market_history(
    request: Request,  # The decorator needs this
    symbol: str,
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=1000, gt=0, le=5000),
) -> List[HistoryDataPoint]:

    # 1. Fast Lookup
    asset_id = await session.scalar(
        select(Asset.id).where(Asset.symbol == symbol.upper())
    )

    if not asset_id:
        raise HTTPException(
            status_code=404, detail=f"Asset symbol '{symbol}' not found."
        )

    # 2. Optimized Query (SQLAlchemy Core)
    stmt = (
        select(MarketDataDaily.__table__, FeaturesDaily.__table__)
        .join(
            FeaturesDaily.__table__,
            (FeaturesDaily.asset_id == MarketDataDaily.asset_id)
            & (FeaturesDaily.time == MarketDataDaily.time),
        )
        .where(MarketDataDaily.asset_id == asset_id)
        .order_by(desc(MarketDataDaily.time))
        .limit(limit)
    )

    result = await session.execute(stmt)
    # Fetch as mappings (dicts) to avoid overhead of SQLAlchemy Objects
    rows = result.mappings().all()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No history found for {symbol}")

    # 3. Construct Pydantic Models
    # Since we defined `model_config = ConfigDict(from_attributes=True)`
    # in our schemas, we can pass these dictionaries directly.
    response_data = []

    for row in rows:
        # Pydantic v2 is smart enough to extract fields from a dict
        # even if they are flat in the row but nested in the schema.

        # However, for cleaner explicit nesting, we construct the sub-objects:
        try:
            # Construct nested FeatureSet
            feature_set = FeatureSet(
                trend=TrendFeatures.model_validate(row),
                momentum=MomentumFeatures.model_validate(row),
                volatility=VolatilityFeatures.model_validate(row),
                volume=VolumeFeatures.model_validate(row),
            )

            # Construct main point
            point = HistoryDataPoint(
                time=row["time"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                features=feature_set,
            )
            response_data.append(point)

        except Exception as e:
            # Log this in production
            print(f"Skipping row due to validation error: {e}")
            continue

    return response_data


@router.get(
    "/snapshots", response_model=List[MarketSnapshot], summary="Get Market Snapshots"
)
async def get_snapshots(
    symbols: str = Query(
        ..., description="Comma-separated list of symbols (e.g., SPY,BTC,NVDA)"
    ),
    lookback_days: int = Query(7, ge=2, le=30),
    session: AsyncSession = Depends(get_session),
):
    """
    Returns the latest price, change, and a sparkline trend for a list of symbols.
    Optimized for dashboards and watchlists.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    if not symbol_list:
        return []

    # 1. Resolve IDs
    asset_stmt = select(Asset.id, Asset.symbol).where(Asset.symbol.in_(symbol_list))
    asset_res = await session.execute(asset_stmt)
    asset_map = {row.symbol: row.id for row in asset_res.all()}

    results = []

    # 2. Fetch Data for each symbol
    # Note: We could do a complex Window Function query to get top N for each ID in one go,
    # but for a watchlist of < 20 items, a simple loop with indexed queries is often faster
    # and much easier to maintain than advanced SQL partition logic.

    for symbol in symbol_list:
        asset_id = asset_map.get(symbol)
        if not asset_id:
            continue

        # Get last N rows
        stmt = (
            select(MarketDataDaily.time, MarketDataDaily.close)
            .where(MarketDataDaily.asset_id == asset_id)
            .order_by(desc(MarketDataDaily.time))
            .limit(
                lookback_days + 1
            )  # +1 to calculate change for the oldest point if needed
        )
        data_res = await session.execute(stmt)
        # Reverse to get chronological order [Oldest -> Newest]
        rows = data_res.all()[::-1]

        if not rows:
            continue

        latest = rows[-1]
        prev = rows[-2] if len(rows) > 1 else None

        # Calculate Change (Today vs Yesterday)
        change = 0.0
        pct_change = 0.0

        if prev:
            change = latest.close - prev.close
            pct_change = change / prev.close

        # Sparkline (List of closes)
        sparkline = [r.close for r in rows]

        results.append(
            MarketSnapshot(
                symbol=symbol,
                price=latest.close,
                change_1d=change,
                change_1d_pct=pct_change,
                last_updated=latest.time,
                sparkline=sparkline,
            )
        )

    return results
