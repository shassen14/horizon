# apps/api_server/routers/assets.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api_server.core.limiter import limiter
from apps.api_server.dependencies.database import get_session
from apps.api_server.schemas.assets import AssetInfo, AssetDetail
from packages.database.models import Asset

router = APIRouter(prefix="/public/assets", tags=["Assets"])


@router.get("", response_model=List[AssetInfo], summary="List or Search Assets")
@limiter.limit("100/minute")
async def list_assets(
    request: Request,  # Required for limiter
    q: Optional[str] = Query(
        None, min_length=1, max_length=20, description="Search query for symbol or name"
    ),
    limit: int = Query(50, le=100),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Asset).where(Asset.is_active == True)

    if q:
        # Simple case-insensitive search on symbol or name
        search_term = f"%{q.upper()}%"
        stmt = stmt.where(
            (Asset.symbol.ilike(search_term)) | (Asset.name.ilike(search_term))
        )

    stmt = stmt.limit(limit)

    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/{symbol}", response_model=AssetDetail, summary="Get Asset Details")
@limiter.limit("100/minute")
async def get_asset_detail(
    request: Request,  # Required for limiter
    symbol: str,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Asset).where(Asset.symbol == symbol.upper())
    result = await session.execute(stmt)
    asset = result.scalar_one_or_none()

    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    return asset
