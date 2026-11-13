import datetime
from decimal import Decimal

import pytest
from psycopg_pool import AsyncConnectionPool

from horizon.db.market_data import insert_bars, latest_times
from horizon.db.types import AssetId, DailyKey, NewBar, Source


async def create_asset(db: AsyncConnectionPool, symbol: str) -> AssetId:
    # Bars reference assets(id), so each test needs a real asset row first.
    async with db.connection() as conn:
        cur = await conn.execute(
            "INSERT INTO assets (symbol) VALUES (%s) RETURNING id", (symbol,)
        )
        row = await cur.fetchone()

    assert row is not None
    return AssetId(row[0])


def make_bar(asset_id: AssetId, date: datetime.date, vwap: Decimal | None) -> NewBar:
    return NewBar(
        time=DailyKey.from_trading_date(date),
        asset_id=asset_id,
        open=Decimal("10.50"),
        high=Decimal("11.25"),
        low=Decimal("10.00"),
        close=Decimal("11.00"),
        volume=1_000,
        vwap=vwap,
        source=Source.ALPACA,
    )


@pytest.mark.asyncio
async def test_db_has_tables(db: AsyncConnectionPool) -> None:
    async with db.connection() as conn:
        cur = await conn.execute("SELECT to_regclass('market_data_daily')")
        table = await cur.fetchone()
        assert table != (None,)


@pytest.mark.asyncio
async def test_latest_times_without_bars_is_empty(db: AsyncConnectionPool) -> None:
    result = await latest_times(db, [AssetId(1)])
    assert result == {}


@pytest.mark.asyncio
async def test_insert_bars_is_idempotent(db: AsyncConnectionPool) -> None:
    asset_id = await create_asset(db, "AAPL")
    bars = [
        make_bar(asset_id, datetime.date(2024, 1, 2), Decimal("10.75")),
        make_bar(asset_id, datetime.date(2024, 1, 3), None),
        make_bar(asset_id, datetime.date(2024, 1, 4), Decimal("10.90")),
    ]

    assert await insert_bars(db, bars) == 3
    # Re-inserting the same bars must skip them all, not duplicate or fail.
    assert await insert_bars(db, bars) == 0

    assert await latest_times(db, [asset_id]) == {asset_id: bars[-1].time}


@pytest.mark.asyncio
async def test_insert_bars_empty_is_noop(db: AsyncConnectionPool) -> None:
    assert await insert_bars(db, []) == 0
