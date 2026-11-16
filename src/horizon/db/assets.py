from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from horizon.db.types import Asset, AssetId, NewAsset


# The current universe: the assets the daily job fetches.
async def active_assets(pool: AsyncConnectionPool) -> list[Asset]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT id, symbol, name, exchange, asset_type, is_active
            FROM assets
            WHERE is_active = true
            ORDER BY symbol
            """,
        )

        rows = await cur.fetchall()

    return [
        Asset(
            id=AssetId(asset_id),
            symbol=symbol,
            name=name,
            exchange=exchange,
            asset_type=asset_type,
            is_active=is_active,
        )
        for asset_id, symbol, name, exchange, asset_type, is_active in rows
    ]


async def upsert_assets(pool: AsyncConnectionPool, assets: list[NewAsset]) -> int:
    async with pool.connection() as conn:
        return await _upsert_assets(conn, assets)


# Replaces the universe in one transaction: deactivate everything, then
# reactivate or insert the fresh list. Assets that drop out keep their rows and
# history with is_active = false, and readers never see the half-done state.
async def refresh_screener(pool: AsyncConnectionPool, fresh: list[NewAsset]) -> int:
    # An empty list almost certainly means the screener failed, not that the
    # market has no stocks; applying it would deactivate everything.
    if not fresh:
        raise ValueError("refusing to deactivate every asset: screener returned none")

    async with pool.connection() as conn, conn.transaction():
        await conn.execute("UPDATE assets SET is_active = false")

        return await _upsert_assets(conn, fresh)


# Takes a connection rather than the pool, so refresh_screener can run it inside
# its own transaction. Matching on symbol keeps an existing asset's id, and with
# it the asset's bars, while refreshing its metadata.
async def _upsert_assets(conn: AsyncConnection, assets: list[NewAsset]) -> int:
    if not assets:
        return 0

    symbols = [asset.symbol for asset in assets]
    names = [asset.name for asset in assets]
    exchanges = [asset.exchange for asset in assets]
    asset_types = [asset.asset_type for asset in assets]

    cur = await conn.execute(
        """
        INSERT INTO assets (symbol, name, exchange, asset_type, is_active)
        SELECT symbol, name, exchange, asset_type, true
        FROM UNNEST (%s::text[], %s::text[], %s::text[], %s::text[])
            AS t(symbol, name, exchange, asset_type)
        ON CONFLICT (symbol) DO UPDATE SET
            name = EXCLUDED.name,
            exchange = EXCLUDED.exchange,
            asset_type = EXCLUDED.asset_type,
            is_active = true
        """,
        (
            symbols,
            names,
            exchanges,
            asset_types,
        ),
    )

    return cur.rowcount
