# packages/ml_core/dataset.py

import polars as pl
from packages.quant_lib.config import settings
from urllib.parse import quote_plus


# This function is now pure. It just needs a logger to talk.
async def load_training_data(start_date: str, end_date: str, logger) -> pl.DataFrame:
    """
    Loads features and market data, then calculates the future return target.
    """
    TARGET_HORIZON_DAYS = 63

    logger.info(f"Loading training data from {start_date} to {end_date}...")

    # URL-encode the password to handle special characters
    safe_password = quote_plus(settings.db.password)

    # Construct the string manually for connectorx
    db_url = (
        f"postgresql://{settings.db.user}:{safe_password}@"
        f"{settings.db.host}:{settings.db.port}/{settings.db.name}"
    )

    # 1. SQL Query to fetch combined feature and market data
    query = f"""
        SELECT 
            fd.*, 
            mdd.close as close_price
        FROM features_daily fd
        JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
        WHERE fd.time >= '{start_date}' AND fd.time <= '{end_date}'
        ORDER BY fd.asset_id, fd.time ASC
    """

    try:
        # Load data directly into Polars
        df = pl.read_database_uri(query=query, uri=db_url)
    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        return pl.DataFrame()

    if df.is_empty():
        logger.warning("No training data found for the specified date range.")
        return pl.DataFrame()

    logger.info(f"Loaded {len(df):,} rows from the database.")

    # 2. Calculate the Target Variable (Forward Return)
    # This is the "secret sauce" of a supervised learning model for finance.
    # We want to predict the return 'TARGET_HORIZON_DAYS' into the future.

    # We use a window function (`shift`) grouped by `asset_id` to prevent looking
    # from one stock's data into another's.
    forward_return_expr = (
        (
            (pl.col("close_price").shift(-TARGET_HORIZON_DAYS) / pl.col("close_price"))
            - 1
        )
        .over("asset_id")
        .alias("target_forward_return")
    )

    df = df.with_columns(forward_return_expr)

    # 3. Data Cleaning
    # The last `TARGET_HORIZON_DAYS` rows for each stock will have a null target,
    # as their future is unknown. We must remove these for training.
    # We also drop rows with null features (the initial warm-up period).

    # Identify feature columns (all columns except identifiers and target)
    feature_cols = [
        col
        for col in df.columns
        if col not in ["time", "asset_id", "close_price", "target_forward_return"]
    ]

    # Drop rows that have nulls in either the target or any of the feature columns
    df_clean = df.drop_nulls(subset=feature_cols + ["target_forward_return"])

    logger.success(f"Dataset preparation complete. Final shape: {df_clean.shape}")

    return df_clean


async def load_regime_data(start_date: str, end_date: str, logger) -> pl.DataFrame:
    logger.info(f"Loading market-wide data for Regime model...")
    safe_password = quote_plus(settings.db.password)
    db_url = (
        f"postgresql://{settings.db.user}:{safe_password}@"
        f"{settings.db.host}:{settings.db.port}/{settings.db.name}"
    )

    # 1. Query all necessary features and close prices in one go
    # This query is much simpler and more direct.
    query = f"""
        SELECT 
            fd.time, fd.asset_id, fd.sma_50, fd.atr_14_pct, fd.rsi_14, fd.zscore_20,
            mdd.close as close_price,
            a.symbol
        FROM features_daily fd
        JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
        JOIN asset_metadata a ON fd.asset_id = a.id
        WHERE fd.time >= '{start_date}' AND fd.time <= '{end_date}'
    """

    try:
        full_df = pl.read_database_uri(query=query, uri=db_url)
    except Exception as e:
        logger.error(f"Failed to load data for regime model: {e}")
        return pl.DataFrame()

    if full_df.is_empty():
        logger.warning("No data found for regime model.")
        return pl.DataFrame()

    logger.info(f"Loaded {len(full_df):,} base rows for regime calculation.")

    # 2. Calculate daily aggregates (Breadth)
    market_df = (
        full_df.group_by("time")
        .agg(
            [
                (pl.col("close_price") > pl.col("sma_50")).sum().cast(pl.Float64)
                / pl.count(),  # breadth_sma50_pct
                pl.col("atr_14_pct").mean().alias("vol_market_avg_atr_pct"),
            ]
        )
        .sort("time")
    )

    # 3. Get SPY features and join them in
    spy_df = (
        full_df.filter(pl.col("symbol") == "SPY")
        .select(["time", "rsi_14", "zscore_20"])
        .rename({"rsi_14": "spy_rsi_14", "zscore_20": "spy_zscore_20"})
    )

    final_df = market_df.join(spy_df, on="time", how="left")

    # 4. Create Target: Did SPY go up in the next 3 months?
    # We calculate the forward return for SPY specifically.
    spy_forward_return = (
        full_df.filter(pl.col("symbol") == "SPY")
        .sort("time")
        .with_columns(
            ((pl.col("close_price").shift(-63) / pl.col("close_price")) - 1).alias(
                "spy_forward_return"
            )
        )
        .select(["time", "spy_forward_return"])
    )

    final_df = final_df.join(spy_forward_return, on="time", how="left")

    # Convert regression target to classification target
    final_df = final_df.with_columns(
        (pl.col("spy_forward_return") > 0.0).cast(pl.Int32).alias("target_regime_bull")
    ).drop("spy_forward_return")

    logger.success(f"Regime dataset preparation complete. Shape: {final_df.shape}")

    return final_df.drop_nulls()
