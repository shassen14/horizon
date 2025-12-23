# apps/api_server/routers/system.py
import os
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta, timezone

from apps.api_server.dependencies.database import get_session
from apps.api_server.schemas.system import SystemStatus
from packages.database.models import MarketDataDaily, MarketData5Min, Asset
from apps.api_server.schemas.enums import SystemHealth, Environment
from packages.quant_lib.config import settings

router = APIRouter(prefix="/public/system", tags=["System Status"])


@router.get("/status", response_model=SystemStatus)
async def get_system_status(session: AsyncSession = Depends(get_session)):
    """
    Checks the freshness of data and overall system health.
    """
    # 1. Get Asset Count
    count_res = await session.execute(
        select(func.count(Asset.id)).where(Asset.is_active == True)
    )
    active_count = count_res.scalar() or 0

    # 2. Get Last Updates (Fastest way is querying the ledger metadata if available,
    # but querying MAX(time) on indexed columns is also reasonably fast on partitioned tables)

    # We use the new Asset ledger columns for speed if populated,
    # or fallback to table max if you haven't run the ledger update yet.
    # For robustness, let's query the max time from the actual data tables
    # (Postgres can optimize MAX() on indexed columns to be instant).

    daily_res = await session.execute(func.max(MarketDataDaily.time))
    last_daily = daily_res.scalar()

    intraday_res = await session.execute(func.max(MarketData5Min.time))
    last_intraday = intraday_res.scalar()

    # 3. Determine Health
    # Logic: If data is older than 24h (on a weekday), system is 'Stale'.
    status = SystemHealth.HEALTHY
    now = datetime.now(timezone.utc)

    # Simple heuristic: If last daily data is > 2 days old, mark stale
    # (accounts for weekends + delayed run)
    if last_daily and (now - last_daily > timedelta(days=2)):
        status = SystemHealth.STALE

    # Determine Environment
    env_str = settings.system.environment

    # Gracefully handle string mapping, default to PRODUCTION if unknown string found
    try:
        current_env = Environment(env_str)
    except ValueError:
        current_env = Environment.PRODUCTION

    return SystemStatus(
        status=status,
        last_daily_update=last_daily,
        last_intraday_update=last_intraday,
        active_assets=active_count,
        environment=current_env,
    )
