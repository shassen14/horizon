import pytest
from psycopg_pool import AsyncConnectionPool

from horizon.db.market_data import latest_times
from horizon.db.types import AssetId


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
