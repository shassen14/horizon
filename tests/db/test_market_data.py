import pytest
from psycopg_pool import AsyncConnectionPool


@pytest.mark.asyncio
async def test_db_has_tables(db: AsyncConnectionPool) -> None:
    async with db.connection() as conn:
        cur = await conn.execute("SELECT to_regclass('market_data_daily')")
        table = await cur.fetchone()
        assert table != (None,)
