import datetime

from psycopg_pool import AsyncConnectionPool

from horizon.db.types import AssetId, DailyKey, NewBar


async def latest_times(
    pool: AsyncConnectionPool, asset_ids: list[AssetId]
) -> dict[AssetId, DailyKey]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT asset_id, MAX(time) AS max_time
            FROM market_data_daily
            WHERE asset_id = ANY(%s)
            GROUP BY asset_id
            """,
            (asset_ids,),
        )

        rows = await cur.fetchall()

    return {
        asset_id: DailyKey(max_time.astimezone(datetime.UTC))
        for asset_id, max_time in rows
    }


async def insert_bars(pool: AsyncConnectionPool, bars: list[NewBar]) -> int:
    if not bars:
        return 0

    times = [bar.time.time for bar in bars]
    asset_ids = [bar.asset_id for bar in bars]
    opens = [bar.open for bar in bars]
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    vwaps = [bar.vwap for bar in bars]
    sources = [bar.source.value for bar in bars]

    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            INSERT INTO market_data_daily (time, asset_id, open, high, low, close, volume, vwap, source, adjustment)
            SELECT time, asset_id, open, high, low, close, volume, vwap, source, 'none'
            FROM UNNEST(
                %s::timestamptz[], %s::int4[], %s::numeric[], %s::numeric[], %s::numeric[], %s::numeric[], %s::int8[], %s::numeric[], %s::text[])
            AS t(time, asset_id, open, high, low, close, volume, vwap, source)
            ON CONFLICT (time, asset_id) DO NOTHING
            """,
            (times, asset_ids, opens, highs, lows, closes, volumes, vwaps, sources),
        )

    return cur.rowcount
