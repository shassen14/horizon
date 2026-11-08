import psycopg


def test_migrated_db_has_tables(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        table = conn.execute("SELECT to_regclass('market_data_daily')").fetchone()
        assert table != (None,)
