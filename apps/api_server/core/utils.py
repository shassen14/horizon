# apps/api_server/core/utils.py

from apps.api_server.schemas.enums import MarketInterval


def interval_to_minutes(interval: MarketInterval) -> int:
    """Converts a MarketInterval enum to integer minutes."""
    value = interval.value  # e.g. "15m", "4h", "1d"

    if value.endswith("m"):
        return int(value[:-1])
    elif value.endswith("h"):
        return int(value[:-1]) * 60
    elif value.endswith("d"):
        return int(value[:-1]) * 1440
    elif value.endswith("w"):
        return int(value[:-1]) * 10080

    raise ValueError(f"Unknown interval format: {value}")
