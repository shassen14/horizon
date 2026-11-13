import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from horizon.db.assets import active_assets, refresh_screener, upsert_assets
from horizon.db.types import NewAsset


def make_asset(symbol: str, name: str | None = None) -> NewAsset:
    return NewAsset(symbol=symbol, name=name, exchange="NASDAQ", asset_type="stock")


async def active_symbols(db: AsyncConnectionPool) -> list[str]:
    return [asset.symbol for asset in await active_assets(db)]


@pytest.mark.asyncio
async def test_active_assets_without_assets_is_empty(db: AsyncConnectionPool) -> None:
    assert await active_assets(db) == []


@pytest.mark.asyncio
async def test_upsert_assets_updates_existing_symbol(db: AsyncConnectionPool) -> None:
    assert await upsert_assets(db, [make_asset("AAPL", "Apple")]) == 1
    # The same symbol again must update the existing row, not add a second one.
    assert await upsert_assets(db, [make_asset("AAPL", "Apple Inc.")]) == 1

    assets = await active_assets(db)
    assert [(a.symbol, a.name) for a in assets] == [("AAPL", "Apple Inc.")]


@pytest.mark.asyncio
async def test_upsert_assets_empty_is_noop(db: AsyncConnectionPool) -> None:
    assert await upsert_assets(db, []) == 0


@pytest.mark.asyncio
async def test_active_assets_is_ordered_by_symbol(db: AsyncConnectionPool) -> None:
    await upsert_assets(db, [make_asset("MSFT"), make_asset("AAPL")])
    assert await active_symbols(db) == ["AAPL", "MSFT"]


@pytest.mark.asyncio
async def test_refresh_screener_deactivates_dropped_symbols(
    db: AsyncConnectionPool,
) -> None:
    await upsert_assets(db, [make_asset("AAPL"), make_asset("MSFT")])

    assert await refresh_screener(db, [make_asset("AAPL")]) == 1
    assert await active_symbols(db) == ["AAPL"]

    # A dropped symbol keeps its row (and so its price history), only inactive.
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT is_active FROM assets WHERE symbol = %s", ("MSFT",)
        )
        assert await cur.fetchone() == (False,)


@pytest.mark.asyncio
async def test_refresh_screener_rolls_back_on_failure(db: AsyncConnectionPool) -> None:
    await upsert_assets(db, [make_asset("AAPL")])

    # A duplicate symbol makes the upsert fail after the deactivating UPDATE
    # already ran. Without a transaction, AAPL would be left inactive.
    with pytest.raises(psycopg.errors.CardinalityViolation):
        await refresh_screener(db, [make_asset("MSFT"), make_asset("MSFT")])

    assert await active_symbols(db) == ["AAPL"]


@pytest.mark.asyncio
async def test_refresh_screener_rejects_empty_universe(
    db: AsyncConnectionPool,
) -> None:
    await upsert_assets(db, [make_asset("AAPL")])

    with pytest.raises(ValueError):
        await refresh_screener(db, [])

    assert await active_symbols(db) == ["AAPL"]
