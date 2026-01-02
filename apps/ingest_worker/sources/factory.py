# apps/ingest_worker/sources/factory.py

from packages.quant_lib.interfaces import DataSource
from .alpaca import AlpacaSource
from .yfinance_source import YFinanceSource
from packages.quant_lib.logging import LogManager


def get_data_source(source_name: str) -> DataSource:
    """Factory to instantiate data sources based on CLI argument."""

    log_manager = LogManager(service_name="ingest-factory")

    if source_name == "alpaca":
        source_logger = log_manager.get_logger("alpaca-source")
        return AlpacaSource(logger=source_logger)

    elif source_name == "yfinance":
        return YFinanceSource()

    else:
        raise ValueError(f"Unknown data source: {source_name}")
