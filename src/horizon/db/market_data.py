import datetime

from psycopg_pool import AsyncConnectionPool

from horizon.db.types import AssetId, DailyKey


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
