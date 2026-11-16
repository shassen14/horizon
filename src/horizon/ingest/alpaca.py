import datetime
import json
from decimal import Decimal

import httpx
from pydantic import BaseModel, ConfigDict, Field


# One bar as /v2/stocks/bars returns it. Aliases map Alpaca's one-letter keys to
# readable names, and validate_by_name lets tests build bars by those names.
# Keys not listed here, such as n (trade count), are ignored.
class AlpacaBar(BaseModel):
    model_config = ConfigDict(frozen=True, validate_by_name=True)

    open: Decimal = Field(alias="o")
    high: Decimal = Field(alias="h")
    low: Decimal = Field(alias="l")
    close: Decimal = Field(alias="c")
    volume: int = Field(alias="v")
    vwap: Decimal | None = Field(default=None, alias="vw")
    # Midnight New York time, expressed in UTC (05:00Z in winter, 04:00Z in
    # summer). DailyKey.from_et_instant turns it into the trading date.
    time: datetime.datetime = Field(alias="t")


# One page of a response; next_page_token is None on the last page.
class BarsPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    bars: dict[str, list[AlpacaBar]]
    next_page_token: str | None


# Daily only for now; timeframe becomes a parameter once a second timeframe has a
# caller. Both start and end dates are inclusive.
async def fetch_daily_bars(
    client: httpx.AsyncClient,
    symbols: list[str],
    start: datetime.date,
    end: datetime.date,
) -> dict[str, list[AlpacaBar]]:
    params: dict[str, str | int] = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        # Unadjusted prices; splits and dividends are stored separately.
        "adjustment": "raw",
        # All US exchanges. The free plan delays it by 15 minutes, which a daily
        # job never notices; iex alone carries only a few percent of volume.
        "feed": "sip",
        # The maximum, counted across all symbols combined, not per symbol.
        "limit": 10_000,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }

    result: dict[str, list[AlpacaBar]] = {}

    while True:
        response = await client.get("/v2/stocks/bars", params=params)

        # Before parsing, so an error response raises with its status code
        # rather than failing as malformed JSON.
        response.raise_for_status()

        # parse_float=Decimal keeps prices exact; as floats they would already
        # have lost precision before pydantic saw them.
        page = BarsPage.model_validate(json.loads(response.text, parse_float=Decimal))

        # A symbol's bars can continue on the next page, so extend, never replace.
        for symbol, bars in page.bars.items():
            result.setdefault(symbol, []).extend(bars)

        if page.next_page_token is None:
            break

        params["page_token"] = page.next_page_token

    return result
