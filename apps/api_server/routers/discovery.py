# apps/api_server/routers/discovery.py

from typing import List, Literal
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api_server.core.limiter import limiter
from apps.api_server.dependencies.database import get_session

from apps.api_server.schemas.discovery import ScreenerResult
from packages.database.models import Asset, FeaturesDaily, MarketDataDaily

router = APIRouter(prefix="/public/discovery", tags=["Discovery"])

# Define allowed sort columns to prevent SQL injection and add new ones
SortableColumn = Literal[
    "relative_volume",
    "rsi_14",
    "return_1d",
    "volume_adv_20",
    "atr_14_pct",
]


@router.get(
    "/market-leaders",
    response_model=List[ScreenerResult],
    operation_id="get_market_leaders",
)
@limiter.limit("60/minute")
async def get_market_leaders(
    request: Request,
    session: AsyncSession = Depends(get_session),
    sort_by: SortableColumn = Query(
        "relative_volume", description="Metric to sort results by."
    ),
    sort_dir: Literal["asc", "desc"] = Query("desc", description="Sort direction."),
    limit: int = Query(50, le=200),
    min_price: float = Query(5.0, ge=0),
    min_avg_volume: int = Query(1_000_000, ge=0),
):
    """
    Provides a ranked list of assets based on key daily metrics, designed for market discovery.
    """

    # 2. Main Query - Expanded to include new fields
    stmt = (
        select(
            Asset.symbol,
            Asset.name,
            MarketDataDaily.close.label("last_price"),
            FeaturesDaily.relative_volume,
            FeaturesDaily.rsi_14,
            FeaturesDaily.return_1d,
            FeaturesDaily.sma_50,
            FeaturesDaily.atr_14_pct,
        )
        # 1. Start with Active Assets (Smallest set: ~3,200 rows)
        .select_from(Asset).where(
            (Asset.is_active == True)
            & (Asset.last_market_data_daily_update.is_not(None))  # Ensure we have data
        )
        # 2. Join Features directly using the Ledger Timestamp
        # This uses the Primary Key Index (time, asset_id) -> Instant Lookup
        .join(
            FeaturesDaily,
            (FeaturesDaily.asset_id == Asset.id)
            & (FeaturesDaily.time == Asset.last_market_data_daily_update),
        )
        # 3. Join Market Data using the same Ledger Timestamp
        .join(
            MarketDataDaily,
            (MarketDataDaily.asset_id == Asset.id)
            & (MarketDataDaily.time == Asset.last_market_data_daily_update),
        )
        # 4. Apply Filters
        .where(
            (MarketDataDaily.close >= min_price)
            & (FeaturesDaily.volume_adv_20 >= min_avg_volume)
        )
    )

    # Apply dynamic sorting
    sort_column = getattr(FeaturesDaily, sort_by)
    if sort_dir == "desc":
        stmt = stmt.order_by(desc(sort_column))
    else:
        stmt = stmt.order_by(sort_column)  # asc is default

    stmt = stmt.limit(limit)

    result = await session.execute(stmt)
    rows = result.mappings().all()

    # 3. Format Response using the new ScreenerResult schema
    response = []
    for i, row in enumerate(rows):

        # Calculate derived fields safely
        sma_50_pct_diff = None
        if row.get("last_price") and row.get("sma_50") and row["sma_50"] > 0:
            sma_50_pct_diff = (row["last_price"] / row["sma_50"]) - 1

        daily_change_pct = (
            (row["return_1d"] * 100) if row.get("return_1d") is not None else 0.0
        )

        response.append(
            ScreenerResult(
                rank=i + 1,
                symbol=row["symbol"],
                name=row["name"],
                latest_price=row["last_price"],
                daily_change_pct=daily_change_pct,
                relative_volume=row.get("relative_volume"),
                rsi_14=row.get("rsi_14"),
                sma_50_pct_diff=sma_50_pct_diff,
                atr_14_pct=row.get("atr_14_pct"),
            )
        )

    return response
