# apps/feature-worker/engine.py

import polars as pl
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from packages.quant_lib.config import settings
from packages.quant_lib.features import FeatureFactory
from packages.database.session import get_db_session
from packages.database.models import Asset, FeaturesDaily


class FeatureEngine:
    def __init__(self, logger):
        self.logger = logger
        self.factory = FeatureFactory(settings)  # Instantiate the factory
        self.active_assets_map: dict[int, str] = {}

    async def load_active_assets(self):
        """Loads the map of assets marked as active from the database."""
        async with get_db_session() as session:
            result = await session.execute(
                select(Asset.id, Asset.symbol).where(Asset.is_active == True)
            )
            self.active_assets_map = {row[0]: row[1] for row in result}
        self.logger.info(
            f"Loaded {len(self.active_assets_map)} active assets to process."
        )

    async def run(self):
        """Main entry point to generate features for all active assets."""
        await self.load_active_assets()

        # Load benchmark data (SPY) once for relative calculations
        benchmark_df = await self._load_asset_data_as_polars("SPY")

        for asset_id, symbol in self.active_assets_map.items():
            self.logger.info(
                f"--- Processing features for {symbol} (ID: {asset_id}) ---"
            )

            # 1. Load raw daily data for the asset
            asset_df = await self._load_asset_data_as_polars(symbol, asset_id)
            if asset_df.is_empty():
                self.logger.warning(f"No raw data found for {symbol}. Skipping.")
                continue

            # 2. Generate features using the factory
            try:
                features_df = self.factory.generate_all(asset_df, benchmark_df)
            except Exception as e:
                self.logger.error(
                    f"Failed to generate features for {symbol}: {e}", exc_info=True
                )
                continue

            # 3. Write features to the database
            await self._write_features_to_db(features_df)

        self.logger.success("Feature generation complete for all active assets.")

    async def _load_asset_data_as_polars(
        self, symbol: str, asset_id: int = None
    ) -> pl.DataFrame:
        """Loads all daily market data for a given symbol into a Polars DataFrame."""
        if asset_id is None:
            # If ID not provided, look it up. Needed for benchmark.
            async with get_db_session() as session:
                result = await session.execute(
                    select(Asset.id).where(Asset.symbol == symbol)
                )
                asset_id = result.scalar_one_or_none()

        if not asset_id:
            return pl.DataFrame()

        # Efficiently read from DB using Polars' native connector
        # Note: Polars read_database uses a synchronous connection string
        # db_url = settings.db.URL
        db_url = (
            f"postgresql://{settings.db.user}:{settings.db.password}@"
            f"{settings.db.host}:{settings.db.port}/{settings.db.name}"
        )
        query = f"SELECT * FROM market_data_daily WHERE asset_id = {asset_id} ORDER BY time ASC"

        try:
            df = pl.read_database_uri(query=query, uri=db_url)
            return df
        except Exception as e:
            self.logger.error(f"Failed to load data for {symbol} from DB: {e}")
            return pl.DataFrame()

    async def _write_features_to_db(self, df: pl.DataFrame):
        """Upserts a DataFrame of features into the features_daily table."""
        if df.is_empty():
            return
        
        # 1. Identify valid columns
        model_columns = {c.name for c in FeaturesDaily.__table__.columns}
        df_columns = set(df.columns)
        columns_to_insert = list(model_columns.intersection(df_columns))
        
        # Prepare records
        records = df.select(columns_to_insert).to_dicts()
        
        if not records:
            return
            
        # 2. Dynamic Chunk Sizing (The "Staff Engineer" Fix)
        # POSTGRES LIMIT: 32,767 (Signed 16-bit integer).
        # We use 30,000 to provide a safety buffer.
        POSTGRES_PARAM_LIMIT = 30_000
        num_columns = len(columns_to_insert)
        
        # Calculate safe chunk size: 30,000 / 30 cols = 1,000 rows
        safe_chunk_size = POSTGRES_PARAM_LIMIT // num_columns
        
        # Clamp it to something reasonable (e.g., at least 1, max 10k)
        safe_chunk_size = max(1, min(10_000, safe_chunk_size))
        
        self.logger.debug(f"Dynamic Write: {num_columns} cols -> Chunk size {safe_chunk_size}")

        async with get_db_session() as session:
            # Upsert Statement Setup
            stmt = insert(FeaturesDaily).values(records)
            update_dict = {
                col.name: col for col in stmt.excluded if col.name not in ["time", "asset_id"]
            }
            
            # 3. Execution Loop
            total_records = len(records)
            for i in range(0, total_records, safe_chunk_size):
                chunk = records[i : i + safe_chunk_size]
                
                # We reconstruct the insert statement for the chunk
                # (SQLAlchemy insert objects are reusable but values binding is cleaner this way)
                stmt = insert(FeaturesDaily).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['time', 'asset_id'],
                    set_=update_dict,
                )
                
                await session.execute(stmt)
                await session.commit()
                
        self.logger.success(f"Successfully upserted {len(records)} feature rows (Chunk size: {safe_chunk_size}).")