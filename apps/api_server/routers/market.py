# apps/api_server/routers/market.py
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

# Absolute imports are now clean and predictable
from apps.api_server.core.limiter import limiter
from apps.api_server.dependencies.database import get_session
from apps.api_server.schemas.enums import MarketInterval
from apps.api_server.schemas.market import (
    HistoryDataPoint,
    MarketSnapshot,
)
from apps.api_server.services.market_data import MarketDataService
from packages.database.models import (
    Asset,
    MarketDataDaily,
)


router = APIRouter(prefix="/public/market", tags=["Market Data"])


@router.get(
    "/history/{symbol}",
    response_model=List[HistoryDataPoint],
    operation_id="get_symbol_history",
)
@limiter.limit("100/minute")
async def get_market_history(
    request: Request,
    symbol: str,
    interval: MarketInterval = Query(
        MarketInterval.DAY_1, description="Aggregation interval"
    ),
    start_date: Optional[datetime] = Query(None, description="Start date (UTC)"),
    end_date: Optional[datetime] = Query(None, description="End date (UTC)"),
    limit: int = Query(default=1000, gt=0, le=5000),
    session: AsyncSession = Depends(get_session),
) -> List[HistoryDataPoint]:

    # 1. Initialize Service
    service = MarketDataService(session)

    # 2. Delegate Logic
    data = await service.get_history(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )

    if not data:
        # We return 404 only if it's truly empty/invalid symbol.
        # If date range is just empty, returning [] is technically valid but 404 is friendlier for UI
        raise HTTPException(
            status_code=404, detail=f"No data found for {symbol} with params provided."
        )

    return data


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
