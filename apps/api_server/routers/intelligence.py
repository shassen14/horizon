# apps/api_server/routers/intelligence.py

import math
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, Request
from sqlalchemy import case, func, select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api_server.core.limiter import limiter
from apps.api_server.dependencies.database import get_session
from apps.api_server.schemas.intelligence import MarketRegime, RegimeType, RiskLevel
from packages.database.models import Asset, MarketDataDaily, FeaturesDaily

router = APIRouter(prefix="/public/intelligence", tags=["AI & Regime"])


# Helper function to clean NaNs
def sanitize_float(value: Optional[float], default: float = 0.0) -> float:
    if value is None or math.isnan(value):
        return default
    return value


@router.get("/regime", response_model=MarketRegime, summary="Get Current Market Regime")
@limiter.limit("60/minute")
async def get_market_regime(
    request: Request, session: AsyncSession = Depends(get_session)
):
    """
    Returns the current market regime (Bull/Bear/Risk-On/Off).
    Currently implements a Heuristic Logic (v1): SPY Trend + Volatility.
    """

    # 1. Fetch Benchmark Data (SPY)
    # We need the latest price and features for SPY
    stmt = (
        select(MarketDataDaily, FeaturesDaily)
        .join(Asset, Asset.id == MarketDataDaily.asset_id)
        .where(Asset.symbol == "SPY")
        .where(
            (MarketDataDaily.time == Asset.last_market_data_daily_update)
            & (FeaturesDaily.time == Asset.last_market_data_daily_update)
            & (FeaturesDaily.asset_id == Asset.id)
        )
    )

    result = await session.execute(stmt)
    row = result.first()

    if not row:
        # Fallback if SPY isn't populated yet
        return MarketRegime(
            timestamp=datetime.now(timezone.utc),
            regime=RegimeType.SIDEWAYS,
            risk_signal=RiskLevel.NEUTRAL,
            trend_score=0.5,
            volatility_score=0.5,
            summary="Insufficient data to determine regime.",
        )

    daily, features = row

    # Calculate Market Breadth & Volatility
    # We need to look at the LATEST row for every active asset.
    # We use a subquery to find the max time per asset.

    latest_time_sq = (
        select(
            MarketDataDaily.asset_id, func.max(MarketDataDaily.time).label("max_time")
        )
        .group_by(MarketDataDaily.asset_id)
        .subquery()
    )

    # Aggregate Query
    agg_stmt = (
        select(
            func.count(
                case((MarketDataDaily.close > FeaturesDaily.sma_50, 1), else_=None)
            ).label("stocks_above_sma50"),
            func.count(MarketDataDaily.asset_id).label("total_stocks"),
            func.avg(FeaturesDaily.atr_14_pct).label("avg_volatility"),
        )
        .select_from(Asset)
        .where(
            (Asset.is_active == True)
            & (Asset.last_market_data_daily_update.is_not(None))
        )
        # Join using the Ledger column -> Instant Index Lookup
        .join(
            MarketDataDaily,
            (MarketDataDaily.asset_id == Asset.id)
            & (MarketDataDaily.time == Asset.last_market_data_daily_update),
        )
        .join(
            FeaturesDaily,
            (FeaturesDaily.asset_id == Asset.id)
            & (FeaturesDaily.time == Asset.last_market_data_daily_update),
        )
    )

    agg_result = await session.execute(agg_stmt)
    agg_row = agg_result.one()

    # Calculate Breadth
    stocks_above = agg_row.stocks_above_sma50 or 0
    total = agg_row.total_stocks or 0
    breadth = stocks_above / total if total > 0 else 0.0

    # Sanitize Breadth (just in case)
    breadth = sanitize_float(breadth, 0.5)

    # Sanitize Avg Volatility
    avg_vol = sanitize_float(agg_row.avg_volatility, 0.0)

    # Hueristic for Regime Bull or Bear Market
    # This acts as a placeholder until the ML model is ready.

    # 1. Determine Trend (Price vs SMA 200)
    # If Price > SMA200, we are generally in a Bull market.
    trend_score = 0.5
    regime = RegimeType.SIDEWAYS

    # Sanitize SMA
    spy_sma_200 = sanitize_float(features.sma_200, None)

    if spy_sma_200 is not None:
        if daily.close > spy_sma_200:
            trend_score = 0.8
            regime = RegimeType.BULL
        else:
            trend_score = 0.2
            regime = RegimeType.BEAR

    # 2. Determine Risk (Volatility via ATR or RSI)
    # High Volatility or Oversold/Overbought extremes = Risk Off
    vol_score = 0.5
    risk = RiskLevel.NEUTRAL

    # Normalize RSI (0-100) to a score (0.0-1.0)
    # RSI > 70 or < 30 usually implies heightened risk of reversal
    spy_rsi_14 = sanitize_float(features.rsi_14, 50.0)

    if spy_rsi_14 > 70 or spy_rsi_14 < 30:
        vol_score = 0.8
        risk = RiskLevel.RISK_OFF
    elif 45 <= spy_rsi_14 <= 55:
        vol_score = 0.2
        risk = RiskLevel.RISK_ON
    else:
        risk = RiskLevel.NEUTRAL

    # 3. Refine Logic
    # In a Bull Market (Price > SMA200), we are generally Risk On unless RSI is extreme.
    if regime == RegimeType.BULL and vol_score < 0.7:
        risk = RiskLevel.RISK_ON
    elif regime == RegimeType.BEAR:
        risk = RiskLevel.RISK_OFF

    # 4. Generate Summary
    # Example: "Bull Market (Breadth: 72%). Risk On."
    breadth_desc = "Strong" if breadth > 0.6 else "Weak" if breadth < 0.4 else "Neutral"
    summary = f"Market is in a {regime.value} trend with {breadth_desc} participation. Risk environment is {risk.value}."

    return MarketRegime(
        timestamp=daily.time,
        regime=regime,
        risk_signal=risk,
        trend_score=trend_score,
        volatility_score=vol_score,
        breadth_pct=breadth,
        market_volatility_avg=avg_vol,
        summary=summary,
    )
