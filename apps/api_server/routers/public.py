# apps/api_server/routers/public.py

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

# Absolute imports are now clean and predictable
from apps.api_server.core.limiter import limiter
from apps.api_server.dependencies.database import get_session
from apps.api_server.schemas.history import (
    HistoryDataPoint,
    FeatureSet,
    TrendFeatures,
    MomentumFeatures,
    VolatilityFeatures,
    VolumeFeatures,
)
from packages.database.models import Asset, MarketDataDaily, FeaturesDaily
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

router = APIRouter(prefix="/public", tags=["Public Market Data"])


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
